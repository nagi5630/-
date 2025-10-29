# streamlit_future_trade_patterns_30.py
# 出来高・ギャップ・イベント情報を組み込み（Seriesの真偽判定を安全化）
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="株価未来パターン予測（30） - 修正版", layout="wide")
st.title("📈 株価未来パターン予測（30パターン対応） - 出来高/ギャップ/イベント統合（安全化）")

# ------------------- ユーティリティ -------------------
def safe_last_value(v):
    if isinstance(v, pd.DataFrame):
        if v.empty: return None
        return safe_last_value(v.iloc[-1,0])
    if isinstance(v, pd.Series):
        if v.empty: return None
        return safe_last_value(v.iloc[-1])
    if isinstance(v,(list,tuple,np.ndarray)):
        if len(v)==0: return None
        return safe_last_value(v[-1])
    return v

def format_price(v):
    v = safe_last_value(v)
    if v is None: return "N/A"
    try:
        return f"{float(v):,.2f}"
    except:
        return str(v)

def pct(a,b):
    try:
        a=float(a); b=float(b)
        if a==0: return 0.0
        return abs(a-b)/abs(a)*100.0
    except:
        return 0.0

def calc_sma(series, period):
    return series.rolling(period,min_periods=1).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=1).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta>0,0.0)
    loss = -delta.where(delta<0,0.0)
    avg_gain = gain.rolling(period,min_periods=1).mean()
    avg_loss = loss.rolling(period,min_periods=1).mean()
    rs = avg_gain/(avg_loss.replace(0,np.nan))
    rsi = 100-(100/(1+rs))
    return rsi.fillna(50.0)

def calc_atr(df, period=14):
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high-low).abs(),
        (high-prev_close).abs(),
        (low-prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# ------------------- 出来高 / ギャップ / イベント取得 (安全チェック追加) -------------------
def compute_volume_signals(df, vol_ma_period=20, spike_threshold=2.0):
    if 'Volume' not in df.columns:
        return {'vol_ma': None, 'vol_latest': None, 'vol_spike': False, 'volume_score': 0}
    vol = df['Volume'].astype(float)
    vol_ma = float(calc_sma(vol, vol_ma_period).iloc[-1])
    vol_latest = float(vol.iloc[-1])
    vol_ratio = vol_latest / max(1e-9, vol_ma)
    vol_spike = vol_ratio >= spike_threshold
    volume_score = 0
    if vol_ratio >= spike_threshold:
        volume_score += int(min(10, (vol_ratio - spike_threshold) * 3 + 4))
    elif vol_ratio < 0.7:
        volume_score -= int(min(6, (0.7 - vol_ratio) * 10))
    return {'vol_ma': vol_ma, 'vol_latest': vol_latest, 'vol_spike': vol_spike, 'volume_score': volume_score, 'vol_ratio': vol_ratio}

def compute_gap_signals(df):
    if len(df) < 2:
        return {'gap_amount': 0.0, 'gap_pct': 0.0, 'gap_score': 0}
    prev_close = float(df['Close'].iloc[-2])
    today_open = float(df['Open'].iloc[-1])
    gap_amount = today_open - prev_close
    gap_pct = gap_amount / max(1e-9, prev_close)
    gap_score = 0
    mag = abs(gap_pct)
    if mag >= 0.02:
        gap_score = int(8 * np.sign(gap_pct))
    elif mag >= 0.005:
        gap_score = int(4 * np.sign(gap_pct))
    return {'gap_amount': gap_amount, 'gap_pct': gap_pct, 'gap_score': gap_score}

POS_KEYWORDS = ['good', 'beat', 'surge', 'rise', 'raise', 'positive', 'upgrade', 'beat expectations', 'beat est', 'strong', 'up', 'outperform', 'beat eps']
NEG_KEYWORDS = ['miss', 'down', 'fall', 'drop', 'lower', 'negative', 'downgrade', 'recall', 'lawsuit', 'delay', 'cut', 'weak', 'sell']
UNCERTAIN_KEYWORDS = ['uncertain','volatile','investigation','guidance','revision','could','may','might']

def fetch_news_yf(symbol, limit=10):
    try:
        t = yf.Ticker(symbol)
        raw = getattr(t, 'news', None)
        news = list(raw) if raw else []
        out = []
        for n in news[:limit]:
            title = n.get('title','') if isinstance(n, dict) else str(n)
            link = n.get('link','') if isinstance(n, dict) else ''
            ts = n.get('providerPublishTime', None) if isinstance(n, dict) else None
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts))
                except:
                    dt = None
            else:
                dt = None
            out.append({'title': title, 'link': link, 'time': dt})
        return out
    except Exception:
        return []

def fetch_news_newsapi(query, apikey, limit=10):
    if not apikey:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'pageSize': limit,
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': apikey
        }
        r = requests.get(url, params=params, timeout=8)
        j = r.json()
        out = []
        for a in j.get('articles',[]):
            t = a.get('title','')
            link = a.get('url','')
            published = a.get('publishedAt', None)
            dt = None
            if published:
                try:
                    dt = datetime.fromisoformat(published.replace('Z','+00:00'))
                except:
                    dt = None
            out.append({'title': t, 'link': link, 'time': dt})
        return out
    except Exception:
        return []

def simple_news_sentiment(news_items):
    score = 0
    details = []
    # 安全に長さを判定
    if news_items is None or (hasattr(news_items, '__len__') and len(news_items)==0):
        return 0, []
    for n in news_items:
        t = (n.get('title') or '').lower() if isinstance(n, dict) else str(n).lower()
        s = 0
        for kw in POS_KEYWORDS:
            if kw in t:
                s += 2
        for kw in NEG_KEYWORDS:
            if kw in t:
                s -= 2
        for kw in UNCERTAIN_KEYWORDS:
            if kw in t:
                s -= 1
        s = max(-5, min(5, s))
        score += s
        details.append({'title': n.get('title','') if isinstance(n, dict) else str(n), 'score': s, 'time': n.get('time') if isinstance(n, dict) else None})
    norm = int(max(-10, min(10, score)))
    return norm, details

# ------------------- 未来パターン＋トレード目安（外部シグナルで補正） -------------------
def predict_future_trade_patterns(df, lookback=60, risk_mult=1.0, use_atr=True,
                                  volume_info=None, gap_info=None, news_sentiment=0):
    closes = df['Close'].values.astype(float)
    highs = df['High'].values.astype(float)
    lows = df['Low'].values.astype(float)

    last_close = float(closes[-1])
    recent_c = closes[-lookback:]
    recent_h = highs[-lookback:]
    recent_l = lows[-lookback:]
    recent_high = float(recent_h.max())
    recent_low = float(recent_l.min())
    mid_range = (recent_high + recent_low)/2.0

    sma5 = float(calc_sma(df['Close'],5).iloc[-1])
    sma20 = float(calc_sma(df['Close'],20).iloc[-1])
    sma50 = float(calc_sma(df['Close'],50).iloc[-1])
    ema21 = float(calc_ema(df['Close'],21).iloc[-1])
    rsi = float(calc_rsi(df['Close']).iloc[-1])
    atr = float(calc_atr(df).iloc[-1]) if use_atr else max(1.0, last_close*0.01)

    trend_up = bool(last_close > sma20 and sma20 > sma50)
    trend_dn = bool(last_close < sma20 and sma20 < sma50)

    patterns = []

    def trade_levels(kind: str, target_price: float, basis: float=None):
        basis = float(basis if basis is not None else last_close)
        if kind == '上昇':
            entry = basis * (1 - 0.005)
            sl = entry - atr * 1.5 * risk_mult
            tp = float(target_price) if target_price is not None else entry + atr * 2.5 * risk_mult
        else:
            entry = basis * (1 + 0.005)
            sl = entry + atr * 1.5 * risk_mult
            tp = float(target_price) if target_price is not None else entry - atr * 2.5 * risk_mult
        return float(entry), float(tp), float(sl)

    def adjusted_confidence(base_conf, kind):
        conf = base_conf
        if volume_info is not None and isinstance(volume_info, dict):
            vol_s = volume_info.get('volume_score',0)
            conf += max(-6, min(8, int(vol_s)))
            try:
                last_up = bool(df['Close'].iloc[-1] > df['Open'].iloc[-1])
                if volume_info.get('vol_spike') and ((kind=='上昇' and last_up) or (kind=='下落' and not last_up)):
                    conf += 3
            except Exception:
                pass
        if gap_info is not None and isinstance(gap_info, dict):
            gap_s = gap_info.get('gap_score',0)
            if (gap_s > 0 and kind=='上昇') or (gap_s < 0 and kind=='下落'):
                conf += int(abs(gap_s) * 0.8)
            else:
                conf -= int(abs(gap_s) * 0.6)
        if news_sentiment:
            if (news_sentiment > 0 and kind=='上昇') or (news_sentiment < 0 and kind=='下落'):
                conf += int(min(6, abs(news_sentiment)))
            else:
                conf -= int(min(6, abs(news_sentiment)*0.6))
        conf = max(10, min(95, int(conf)))
        return conf

    def add_pattern(name, kind, base_conf, target=None, basis=None):
        conf = adjusted_confidence(base_conf, kind)
        entry, tp, sl = trade_levels(kind, target, basis)
        rr = abs(tp - entry) / max(1e-9, abs(entry - sl))
        patterns.append({
            'name': name,
            'kind': kind,
            'confidence': int(conf),
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'rr': rr,
            'target': tp
        })

    def local_max(a):
        return a[-2] > a[-3] and a[-2] > a[-1]
    def local_min(a):
        return a[-2] < a[-3] and a[-2] < a[-1]

    a = recent_c

    # w1/w2 を安全に初期化（後で参照しても未定義にならないように）
    width_now = recent_high - recent_low
    w1 = w2 = width_now

    if len(a) >= 3 and local_max(a):
        target = recent_low - (recent_high - recent_low) * 0.6
        add_pattern('ダブルトップ', '下落', 62 + (1 if trend_dn else -1) + (1 if rsi>65 else 0), target)
    if len(a) >= 3 and local_min(a):
        target = recent_high + (recent_high - recent_low) * 0.6
        add_pattern('ダブルボトム', '上昇', 62 + (1 if trend_up else -1) + (1 if rsi<35 else 0), target)

    if len(a) >= 5:
        mid = float(a[-3])
        if mid == max(a[-5:]):
            target = recent_low - (mid - recent_low) * 0.7
            add_pattern('トリプルトップ', '下落', 60, target)
        if mid == min(a[-5:]):
            target = recent_high + (recent_high - mid) * 0.7
            add_pattern('トリプルボトム', '上昇', 60, target)

    if len(a) >= 7:
        mid = float(a[-4])
        if mid == max(a[-7:]) and trend_dn:
            target = recent_low - (mid - recent_low)
            add_pattern('ヘッド＆ショルダー', '下落', 61, target)
        if mid == min(a[-7:]) and trend_up:
            target = recent_high + (recent_high - mid)
            add_pattern('逆ヘッド＆ショルダー', '上昇', 61, target)

    if len(a) >= 10:
        mid = float(a[-5])
        if mid == min(a[-10:]):
            target = recent_high + (recent_high - mid) * 0.8
            add_pattern('ソーサー型', '上昇', 57, target)
        if mid == max(a[-10:]):
            target = recent_low - (mid - recent_low) * 0.8
            add_pattern('ラウンドトップ型', '下落', 57, target)

    if last_close > sma5:
        add_pattern('フラッグ上昇', '上昇', 53, last_close + atr * 2.0)
    else:
        add_pattern('フラッグ下降', '下落', 53, last_close - atr * 2.0)

    if abs(last_close - sma5)/max(1e-9, sma5) < 0.01:
        if trend_up:
            add_pattern('ペナント上昇', '上昇', 52, last_close + atr * 1.8)
        if trend_dn:
            add_pattern('ペナント下降', '下落', 52, last_close - atr * 1.8)

    if len(a) >= 3 and a[-2] < a[-3] and a[-2] < a[-1]:
        add_pattern('V字回復', '上昇', 56, last_close + atr * 2.2)
    if len(a) >= 3 and a[-2] > a[-3] and a[-2] > a[-1]:
        add_pattern('逆V字', '下落', 56, last_close - atr * 2.2)

    if last_close > recent_high * 1.003:
        add_pattern('レンジブレイク上', '上昇', 60, last_close + (recent_high - recent_low))
    if last_close < recent_low * 0.997:
        add_pattern('レンジブレイク下', '下落', 60, last_close - (recent_high - recent_low))

    if abs(last_close - mid_range)/max(1e-9, mid_range) < 0.02:
        add_pattern('レンジ反発', '上昇' if last_close>mid_range else '下落', 51, mid_range)

    slope20 = float(calc_sma(df['Close'],20).iloc[-1] - calc_sma(df['Close'],20).iloc[-5]) if len(df)>=5 else 0.0
    if slope20>0 and last_close>ema21 and (sma20 - sma50) > 0:
        add_pattern('上昇ウェッジ（警戒）', '下落', 54, last_close - atr * 2.0)
    if slope20<0 and last_close<ema21 and (sma20 - sma50) < 0:
        add_pattern('下降ウェッジ（反発予兆）', '上昇', 54, last_close + atr * 2.0)

    # トライアングル幅収束判定（安全に幅を計算）
    try:
        width_prev = (highs[-2*lookback:-lookback].max() - lows[-2*lookback:-lookback].min()) if len(highs) >= 2*lookback else width_now
    except Exception:
        width_prev = width_now
    contracting = width_now < width_prev * 0.8
    if contracting:
        if trend_up:
            add_pattern('上昇トライアングル', '上昇', 58, last_close + width_now)
        if trend_dn:
            add_pattern('下降トライアングル', '下落', 58, last_close - width_now)
        add_pattern('対称トライアングル', '上昇' if trend_up else '下落', 55, last_close + (width_now if trend_up else -width_now))

    dev = last_close - sma20
    if trend_up and dev>0:
        add_pattern('上昇チャネル', '上昇', 57, last_close + atr * 2.0)
    if trend_dn and dev<0:
        add_pattern('下降チャネル', '下落', 57, last_close - atr * 2.0)

    if len(a)>=15:
        mid15 = float(a[-8])
        if mid15 == min(a[-15:]) and trend_up:
            add_pattern('カップ＆ハンドル', '上昇', 59, recent_high + atr * 2.5)

    if len(a) >= 20:
        w1 = (highs[-20:-10].max() - lows[-20:-10].min())
        w2 = (highs[-10:].max() - lows[-10:].min())
        if w1 > 0 and w2 < w1*0.7:
            if trend_up:
                add_pattern('ダイヤモンドトップ', '下落', 55, last_close - w2)
            else:
                add_pattern('ダイヤモンドボトム', '上昇', 55, last_close + w2)

    if len(a) >= 20:
        w1 = (highs[-20:-10].max() - lows[-20:-10].min())
        w2 = (highs[-10:].max() - lows[-10:].min())
        if w2 > w1 * 1.2:
            add_pattern('メガホン型', '下落' if rsi>65 else '上昇', 54, last_close + (atr * ( -2.0 if rsi>65 else 2.0)))

    # 安全に w1,w2 を参照
    try:
        if w2 > w1 and not contracting:
            add_pattern('エクスパンディングトライアングル', '上昇' if trend_up else '下落', 52, last_close + (width_now if trend_up else -width_now))
    except Exception:
        pass

    if abs(last_close - mid_range)/max(1e-9, mid_range) < 0.01 and not contracting:
        add_pattern('ボックスレンジ維持', '上昇' if last_close>mid_range else '下落', 50, mid_range)

    return patterns

# ------------------- UI -------------------
st.sidebar.header("⚙️ オプション")
symbol = st.sidebar.text_input("銘柄コード", "AAPL")
period = st.sidebar.selectbox("期間", ["6mo","1y","2y","5y"], index=1)
lookback = st.sidebar.slider("予測に使う直近本数", min_value=20, max_value=120, value=60, step=5)
risk_mult = st.sidebar.slider("リスク係数 (SL/TPのスケール)", 0.5, 2.0, 1.0, 0.1)
use_atr = st.sidebar.checkbox("ATRを使ってTP/SLを調整", value=True)
show_all = st.sidebar.checkbox("上位だけでなく全パターンを表示", value=False)
newsapi_key = st.sidebar.text_input("（任意）NewsAPI API Key（あれば追加ニュースを取得）", value="")
run = st.sidebar.button("未来予測を実行")

st.markdown("直近データから30種のチャート形状を **未来予測** としてスコアリング。出来高・ギャップ・イベント情報を自動取得して信頼度を補正します。※参考用")

placeholder_table = st.empty()
placeholder_chart = st.empty()
placeholder_events = st.empty()
placeholder_meta = st.empty()

if run:
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False).dropna()
        if df is None or (hasattr(df, 'empty') and df.empty):
            st.error("データ取得できませんでした")
            st.stop()

        # 指標
        df['SMA20'] = calc_sma(df['Close'],20)
        df['SMA50'] = calc_sma(df['Close'],50)
        df['EMA21'] = calc_ema(df['Close'],21)
        df['RSI'] = calc_rsi(df['Close'])
        df['ATR'] = calc_atr(df)

        # 出来高・ギャップ・ニュース取得
        volume_info = compute_volume_signals(df)
        gap_info = compute_gap_signals(df)

        news_yf = fetch_news_yf(symbol, limit=8)
        news_api_items = fetch_news_newsapi(symbol, newsapi_key, limit=5) if newsapi_key else []
        all_news = []
        if isinstance(news_yf, list):
            all_news.extend(news_yf)
        if isinstance(news_api_items, list):
            all_news.extend(news_api_items)

        news_sentiment, news_details = simple_news_sentiment(all_news)

        # 決算カレンダー等（安全に取得）
        earnings_info = {}
        try:
            t = yf.Ticker(symbol)
            cal = getattr(t, 'calendar', None)
            earnings = None
            try:
                earnings = getattr(t, 'earnings_dates', None)
            except Exception:
                earnings = None
            earnings_info['calendar'] = cal
            earnings_info['earnings'] = earnings
        except Exception:
            earnings_info['calendar'] = None
            earnings_info['earnings'] = None

        patterns = predict_future_trade_patterns(df, lookback=lookback, risk_mult=risk_mult, use_atr=use_atr,
                                                 volume_info=volume_info, gap_info=gap_info, news_sentiment=news_sentiment)
        if not patterns:
            st.warning("パターンは検出されませんでした")
        else:
            patterns_sorted = sorted(patterns, key=lambda x: (x.get('confidence',50), x.get('rr',1.0)), reverse=True)
            top = patterns_sorted if show_all else patterns_sorted[:6]

            table_rows = []
            for p in top:
                table_rows.append({
                    'パターン': p['name'],
                    '種別': p['kind'],
                    '信頼度(%)': p['confidence'],
                    'エントリー': format_price(p['entry']),
                    '利確目安': format_price(p['tp']),
                    '損切り': format_price(p['sl']),
                    '想定RR': round(float(p['rr']),2)
                })
            placeholder_table.subheader("🔎 信頼度順の候補（出来高/ギャップ/イベントで補正済）")
            placeholder_table.table(pd.DataFrame(table_rows))

            meta_text = f"最新終値: {format_price(df['Close'].iloc[-1])} | 出来高(最新): {int(volume_info.get('vol_latest',0)) if volume_info.get('vol_latest') is not None else 'N/A'} | Vol比: {round(volume_info.get('vol_ratio',0),2) if 'vol_ratio' in volume_info else 'N/A'} | ギャップ率: {round(gap_info.get('gap_pct',0)*100,3)}%"
            placeholder_meta.info(meta_text)

            with placeholder_events:
                st.subheader("🗂️ 取得イベント & ニュース要約")
                col1, col2 = st.columns([2,1])
                with col1:
                    st.write("**直近ニュース（見出し）**")
                    if all_news and len(all_news)>0 and news_details:
                        for nd in news_details:
                            t = nd.get('title','')
                            s = nd.get('score',0)
                            tag = "↑" if s>0 else ("↓" if s<0 else "・")
                            st.write(f"{tag} ({s}) {t}")
                    else:
                        st.write("ニュースが見つかりませんでした。")
                with col2:
                    st.write("**決算/カレンダー情報 (yfinance)**")
                    cal = earnings_info.get('calendar')
                    if cal is not None:
                        if isinstance(cal, (pd.DataFrame, pd.Series)):
                            if not cal.empty:
                                try:
                                    st.write(cal)
                                except Exception:
                                    st.write("カレンダー情報あり（表示不可）")
                            else:
                                st.write("決算カレンダー情報なし")
                        else:
                            st.write(str(cal))
                    else:
                        st.write("決算カレンダー情報なし")

            # 描画
            fig, (ax_price, ax_vol) = plt.subplots(2,1, figsize=(14,8), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
            ax_price.plot(df.index, df['Close'], label='Close')
            ax_price.plot(df.index, df['SMA20'], label='SMA20')
            ax_price.plot(df.index, df['SMA50'], label='SMA50')
            ax_price.grid(alpha=0.2)
            ax_price.set_title(f"{symbol} - {period} チャートと未来予測（出来高/ギャップ/イベント統合）")

            if 'Volume' in df.columns:
                ax_vol.bar(df.index, df['Volume'])
                ax_vol.set_ylabel('Volume')

            last_x = df.index[-1]
            last_price = float(df['Close'].iloc[-1])

            try:
                g = gap_info
                if isinstance(g, dict) and abs(g.get('gap_pct',0)) > 0.001:
                    ax_price.axvline(df.index[-1], linestyle='--', alpha=0.5)
                    ax_price.annotate(f"Gap {round(g['gap_pct']*100,2)}%", xy=(df.index[-1], last_price),
                                      xytext=(df.index[-1], last_price*(1+ (0.03 if g['gap_pct']>0 else -0.03))),
                                      arrowprops=dict(arrowstyle='-|>', color=('green' if g['gap_pct']>0 else 'red')))

            except Exception:
                pass

            for p in top:
                color = 'green' if p['kind']=='上昇' else 'red'
                try:
                    ax_price.annotate(f"{'↑' if p['kind']=='上昇' else '↓'}{p['name']} ({p['confidence']}%)",
                                      xy=(last_x, last_price),
                                      xytext=(last_x, float(p['tp'])),
                                      arrowprops=dict(facecolor=color, shrink=0.05, width=1.5),
                                      fontsize=9)
                except Exception:
                    pass
                ax_price.axhline(float(p['entry']), linestyle='--', alpha=0.25)
                ax_price.axhline(float(p['tp']), linestyle='-', alpha=0.25)
                ax_price.axhline(float(p['sl']), linestyle=':', alpha=0.25)

            ax_price.legend(loc='upper left', ncol=3, fontsize=8)
            plt.tight_layout()
            placeholder_chart.pyplot(fig)

            st.info("※ 本ツールは参考用です。ニュースは見出しベースの簡易評価のため、詳細は原文確認を推奨します。NewsAPIキーを入れると追加で英文ニュースを取得します。")

    except Exception as e:
        st.error(f"エラー発生: {e}")
