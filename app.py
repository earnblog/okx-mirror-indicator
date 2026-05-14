import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(
    page_title="OKX Mirror Indicator",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0c10; color: #e0e0e0; }
h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; color: #e0e0e0 !important; }
[data-testid="stSidebar"] { background: #0d0f14 !important; border-right: 1px solid #1e2230; }
[data-testid="stSidebar"] * { color: #ccc !important; }
.mirror-badge {
    display: inline-block; background: #1a2a3a; color: #378ADD;
    border: 1px solid #378ADD55; border-radius: 6px;
    padding: 3px 12px; font-size: 12px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; margin-left: 10px;
}
.normal-badge {
    display: inline-block; background: #1a2a1a; color: #1D9E75;
    border: 1px solid #1D9E7555; border-radius: 6px;
    padding: 3px 12px; font-size: 12px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; margin-left: 10px;
}
.matrix-cell {
    text-align: center; padding: 6px 4px; border-radius: 4px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
.hint-box {
    background: #0d1a2a; border: 1px solid #1a3a5a; border-left: 3px solid #378ADD;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 12px; color: #89b4e0; line-height: 1.6;
}
.hint-box-normal {
    background: #0d1a14; border: 1px solid #1a3a2a; border-left: 3px solid #1D9E75;
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    font-size: 12px; color: #7ecfaa; line-height: 1.6;
}
.stat-card {
    background: #111318; border: 1px solid #1e2230; border-radius: 8px;
    padding: 10px 14px; text-align: center;
}
.stat-val { font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 11px; color: #666; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

BASE = "https://www.okx.com/api/v5"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

POPULAR = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
           "BNB-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP",
           "ADA-USDT-SWAP", "AVAX-USDT-SWAP", "LINK-USDT-SWAP", "DOT-USDT-SWAP"]

TIMEFRAMES = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}

# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def fetch_candles(inst_id: str, bar: str, limit: int = 150):
    try:
        url = f"{BASE}/market/candles"
        r = requests.get(url, params={"instId": inst_id, "bar": bar, "limit": limit},
                         headers=HEADERS, timeout=8)
        data = r.json()
        if data.get("code") != "0":
            return pd.DataFrame()
        rows = data["data"]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df = df.astype({"ts": float, "open": float, "high": float, "low": float,
                        "close": float, "vol": float})
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.sort_values("ts").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_instruments():
    try:
        r = requests.get(f"{BASE}/public/instruments",
                         params={"instType": "SWAP"}, headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("code") != "0":
            return []
        return [d["instId"] for d in data["data"] if d["instId"].endswith("USDT-SWAP")]
    except Exception:
        return []

# ── Indicator calculations ────────────────────────────────────────────────────

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    diff = ema_fast - ema_slow
    dea = calc_ema(diff, signal)
    hist = (diff - dea) * 2
    return diff, dea, hist

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def get_macd_state(hist_val, diff_val):
    """Return text state of MACD"""
    if diff_val > 0 and hist_val > 0:
        return "0轴上方 / 扩张", "#1D9E75"
    elif diff_val > 0 and hist_val <= 0:
        return "0轴上方 / 收缩", "#EF9F27"
    elif diff_val <= 0 and hist_val < 0:
        return "0轴下方 / 扩张", "#D85A30"
    else:
        return "0轴下方 / 收缩", "#EF9F27"

def get_rsi_state(rsi_val):
    if rsi_val >= 70:
        return f"超买 {rsi_val:.1f}", "#D85A30"
    elif rsi_val <= 30:
        return f"超卖 {rsi_val:.1f}", "#1D9E75"
    elif rsi_val >= 60:
        return f"偏强 {rsi_val:.1f}", "#EF9F27"
    elif rsi_val <= 40:
        return f"偏弱 {rsi_val:.1f}", "#7F77DD"
    else:
        return f"中性 {rsi_val:.1f}", "#888888"

def mirror_macd_hint(diff_val, hist_val, dea_val):
    hints = []
    if diff_val > 0:
        hints.append("🪞 MACD镜像：DIFF在0轴下方，对应空头趋势")
    else:
        hints.append("🪞 MACD镜像：DIFF在0轴上方，对应多头趋势")
    if hist_val > 0:
        hints.append("柱线翻转后向下，空头动能增强 → 若现实如此，下跌还在加速")
    else:
        hints.append("柱线翻转后向上，空头动能衰减 → 若现实如此，下跌可能放缓")
    cross = diff_val - dea_val
    if abs(cross) < abs(diff_val) * 0.1:
        hints.append("⚡ 接近金叉/死叉临界点，镜像视角下是反向交叉，留意变盘")
    return hints

def mirror_rsi_hint(rsi_val):
    mirrored = 100 - rsi_val
    hints = []
    if mirrored >= 70:
        hints.append(f"🪞 RSI镜像后 = {mirrored:.1f}（超买区）→ 正常图已是超卖，反弹动能充足")
    elif mirrored <= 30:
        hints.append(f"🪞 RSI镜像后 = {mirrored:.1f}（超卖区）→ 正常图已是超买，追多需谨慎")
    else:
        hints.append(f"🪞 RSI镜像后 = {mirrored:.1f}，镜像视角中性区间")
    if rsi_val > 70:
        hints.append("当前多头极度亢奋，镜像下你会犹豫做空吗？这就是多头该有的反思")
    elif rsi_val < 30:
        hints.append("当前空头极度亢奋，镜像下对应超买，问问自己：空头是否已过度")
    return hints

# ── Chart drawing with Plotly ─────────────────────────────────────────────────

def draw_kline(df, ema20, ema52, ema200):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="K线",
        increasing_fillcolor="#1D9E75", increasing_line_color="#1D9E75",
        decreasing_fillcolor="#D85A30", decreasing_line_color="#D85A30",
    ))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema20, name="EMA20",
                             line=dict(color="#E24B4A", width=1.2)))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema52, name="EMA52",
                             line=dict(color="#EF9F27", width=1.2)))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema200, name="EMA200",
                             line=dict(color="#1D9E75", width=1.2)))

    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0d0f14",
        font=dict(color="#aaa", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1e2230", rangeslider_visible=False,
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#1e2230", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=60, t=30, b=10),
        height=280,
    )
    return fig

def draw_macd(df, diff, dea, hist, mirror=False):
    import plotly.graph_objects as go

    x = df["ts"]
    d = diff.values.copy()
    de = dea.values.copy()
    h = hist.values.copy()

    if mirror:
        d = -d
        de = -de
        h = -h

    colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in h]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=h, name="柱", marker_color=colors,
                         marker_line_width=0, opacity=0.85))
    fig.add_trace(go.Scatter(x=x, y=d, name="DIFF",
                             line=dict(color="#378ADD", width=1.4)))
    fig.add_trace(go.Scatter(x=x, y=de, name="DEA",
                             line=dict(color="#E24B4A", width=1.4)))
    fig.add_hline(y=0, line_color="#334", line_width=1, line_dash="dot")

    if mirror:
        fig.add_annotation(x=0.01, y=0.95, xref="paper", yref="paper",
                           text="🪞 MACD 镜像视角", showarrow=False,
                           font=dict(color="#378ADD", size=11),
                           bgcolor="#0d1a2a", bordercolor="#378ADD",
                           borderwidth=1, borderpad=4)

    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0d0f14",
        font=dict(color="#aaa", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1e2230", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#1e2230", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=160,
        barmode="relative",
    )
    return fig

def draw_rsi(df, rsi, mirror=False):
    import plotly.graph_objects as go

    x = df["ts"]
    r = (100 - rsi.values) if mirror else rsi.values

    ob = 70 if not mirror else 30
    os_ = 30 if not mirror else 70

    fig = go.Figure()
    fig.add_hline(y=70, line_color="#D85A30", line_width=0.8,
                  line_dash="dot", annotation_text="70", annotation_position="right")
    fig.add_hline(y=50, line_color="#444", line_width=0.8, line_dash="dot")
    fig.add_hline(y=30, line_color="#1D9E75", line_width=0.8,
                  line_dash="dot", annotation_text="30", annotation_position="right")
    fig.add_trace(go.Scatter(x=x, y=r, name="RSI",
                             line=dict(color="#7F77DD", width=1.5),
                             fill="tozeroy",
                             fillcolor="rgba(127,119,221,0.08)"))

    if mirror:
        fig.add_annotation(x=0.01, y=0.95, xref="paper", yref="paper",
                           text="🪞 RSI 镜像视角", showarrow=False,
                           font=dict(color="#378ADD", size=11),
                           bgcolor="#0d1a2a", bordercolor="#378ADD",
                           borderwidth=1, borderpad=4)

    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0d0f14",
        font=dict(color="#aaa", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1e2230", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#1e2230", side="right",
                   range=[-5, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=130,
    )
    return fig

# ── Matrix ────────────────────────────────────────────────────────────────────

def build_matrix(symbols, timeframe_key, mirror):
    bar = TIMEFRAMES[timeframe_key]
    rows = []
    for sym in symbols:
        df = fetch_candles(sym, bar, 60)
        if df.empty or len(df) < 30:
            rows.append({"币种": sym.replace("-USDT-SWAP", "").replace("-USDT", ""),
                         "MACD状态": "—", "RSI": "—", "macd_color": "#333", "rsi_color": "#333"})
            continue
        close = df["close"]
        diff, dea, hist = calc_macd(close)
        rsi = calc_rsi(close)
        last_diff = diff.iloc[-1]
        last_hist = hist.iloc[-1]
        last_rsi = rsi.iloc[-1]

        if mirror:
            last_diff = -last_diff
            last_hist = -last_hist
            last_rsi = 100 - last_rsi

        macd_txt, macd_col = get_macd_state(last_hist, last_diff)
        rsi_txt, rsi_col = get_rsi_state(last_rsi)
        rows.append({"币种": sym.replace("-USDT-SWAP", "").replace("-USDT", ""),
                     "MACD状态": macd_txt, "RSI": rsi_txt,
                     "macd_color": macd_col, "rsi_color": rsi_col})
    return rows

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🪞 Mirror Indicator")
    st.markdown("---")

    all_instruments = fetch_instruments()
    search_input = st.text_input("搜索交易对", placeholder="BTC, ETH, SOL...")
    if search_input:
        filtered = [i for i in all_instruments if search_input.upper() in i][:20]
        options = filtered if filtered else POPULAR
    else:
        options = POPULAR

    selected_symbol = st.selectbox("交易对", options, index=0)

    tf = st.selectbox("时间周期", list(TIMEFRAMES.keys()),
                      index=3, format_func=lambda x: x)

    st.markdown("---")
    st.markdown("### 🪞 镜像控制")
    macd_mirror = st.toggle("MACD 镜像翻转", value=False)
    rsi_mirror = st.toggle("RSI 镜像翻转", value=False)

    st.markdown("---")
    st.markdown("### 周期矩阵")
    matrix_symbols = st.multiselect(
        "选择对比币种",
        options=POPULAR,
        default=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
    )
    matrix_tf = st.selectbox("矩阵周期", list(TIMEFRAMES.keys()), index=3,
                              key="matrix_tf")

    st.markdown("---")
    auto_refresh = st.toggle("自动刷新 (10秒)", value=False)
    if st.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()

# ── Auto refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────

mode_badge = '<span class="mirror-badge">🪞 镜像模式</span>' if (macd_mirror or rsi_mirror) \
             else '<span class="normal-badge">✓ 正常模式</span>'

st.markdown(f"## {selected_symbol} · {tf} {mode_badge}", unsafe_allow_html=True)

df = fetch_candles(selected_symbol, TIMEFRAMES[tf], 200)

if df.empty:
    st.error("数据加载失败，请稍后重试或检查交易对名称")
    st.stop()

close = df["close"]
ema20 = calc_ema(close, 20)
ema52 = calc_ema(close, 52)
ema200 = calc_ema(close, 200)
diff, dea, hist = calc_macd(close)
rsi = calc_rsi(close)

last_diff = diff.iloc[-1]
last_dea = dea.iloc[-1]
last_hist = hist.iloc[-1]
last_rsi = rsi.iloc[-1]
last_close = close.iloc[-1]
prev_close = close.iloc[-2]
pct_chg = (last_close - prev_close) / prev_close * 100

# ── Stats row ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    chg_color = "#1D9E75" if pct_chg >= 0 else "#D85A30"
    st.markdown(f"""<div class="stat-card">
        <div class="stat-val" style="color:{chg_color}">{last_close:,.4g}</div>
        <div class="stat-label">最新价 ({pct_chg:+.2f}%)</div>
    </div>""", unsafe_allow_html=True)

with c2:
    disp_diff = -last_diff if macd_mirror else last_diff
    diff_color = "#1D9E75" if disp_diff >= 0 else "#D85A30"
    label = "DIFF (镜像)" if macd_mirror else "DIFF"
    st.markdown(f"""<div class="stat-card">
        <div class="stat-val" style="color:{diff_color}">{disp_diff:.4f}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    disp_hist = -last_hist if macd_mirror else last_hist
    hist_color = "#1D9E75" if disp_hist >= 0 else "#D85A30"
    label = "柱线 (镜像)" if macd_mirror else "柱线"
    st.markdown(f"""<div class="stat-card">
        <div class="stat-val" style="color:{hist_color}">{disp_hist:.4f}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    disp_rsi = 100 - last_rsi if rsi_mirror else last_rsi
    rsi_txt, rsi_col = get_rsi_state(disp_rsi)
    label = "RSI (镜像)" if rsi_mirror else "RSI"
    st.markdown(f"""<div class="stat-card">
        <div class="stat-val" style="color:{rsi_col}">{disp_rsi:.1f}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

with c5:
    ema200_val = ema200.iloc[-1]
    above = last_close > ema200_val
    trend_txt = "多头趋势" if above else "空头趋势"
    trend_col = "#1D9E75" if above else "#D85A30"
    if macd_mirror or rsi_mirror:
        trend_txt = "空头趋势(镜)" if above else "多头趋势(镜)"
        trend_col = "#D85A30" if above else "#1D9E75"
    st.markdown(f"""<div class="stat-card">
        <div class="stat-val" style="color:{trend_col}; font-size:15px;">{trend_txt}</div>
        <div class="stat-label">EMA200 位置</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── K线 (always normal) ───────────────────────────────────────────────────────

st.markdown("**K线图** · EMA 20 / 52 / 200（不参与镜像）")
st.plotly_chart(draw_kline(df, ema20, ema52, ema200),
                use_container_width=True, config={"displayModeBar": False})

# ── MACD ─────────────────────────────────────────────────────────────────────

macd_title = "MACD 🪞 镜像视角" if macd_mirror else "MACD 正常视角"
st.markdown(f"**{macd_title}**")
st.plotly_chart(draw_macd(df, diff, dea, hist, macd_mirror),
                use_container_width=True, config={"displayModeBar": False})

if macd_mirror:
    hints = mirror_macd_hint(last_diff, last_hist, last_dea)
    hint_html = "<br>".join(f"· {h}" for h in hints)
    st.markdown(f'<div class="hint-box">{hint_html}</div>', unsafe_allow_html=True)
else:
    macd_state, macd_col = get_macd_state(last_hist, last_diff)
    st.markdown(f'<div class="hint-box-normal">· 当前MACD：<b style="color:{macd_col}">{macd_state}</b> &nbsp;|&nbsp; 开启镜像可切换反向视角</div>',
                unsafe_allow_html=True)

# ── RSI ───────────────────────────────────────────────────────────────────────

rsi_title = "RSI 🪞 镜像视角" if rsi_mirror else "RSI 正常视角"
st.markdown(f"**{rsi_title}**")
st.plotly_chart(draw_rsi(df, rsi, rsi_mirror),
                use_container_width=True, config={"displayModeBar": False})

if rsi_mirror:
    hints = mirror_rsi_hint(last_rsi)
    hint_html = "<br>".join(f"· {h}" for h in hints)
    st.markdown(f'<div class="hint-box">{hint_html}</div>', unsafe_allow_html=True)
else:
    rsi_txt, rsi_col = get_rsi_state(last_rsi)
    st.markdown(f'<div class="hint-box-normal">· 当前RSI：<b style="color:{rsi_col}">{rsi_txt}</b> &nbsp;|&nbsp; 开启镜像可切换反向视角</div>',
                unsafe_allow_html=True)

# ── Multi-symbol Matrix ───────────────────────────────────────────────────────

st.markdown("---")
mirror_label = " 🪞 镜像" if (macd_mirror or rsi_mirror) else ""
st.markdown(f"### 多币种周期矩阵{mirror_label} · {matrix_tf}")

if matrix_symbols:
    mirror_any = macd_mirror or rsi_mirror
    with st.spinner("加载矩阵数据..."):
        rows = build_matrix(matrix_symbols, matrix_tf, mirror_any)

    header_cols = st.columns([1.2] + [1] * len(rows))
    with header_cols[0]:
        st.markdown("<div style='font-size:11px;color:#666;padding:4px'>指标</div>",
                    unsafe_allow_html=True)
    for i, row in enumerate(rows):
        with header_cols[i + 1]:
            st.markdown(f"<div style='font-size:12px;font-weight:700;color:#ccc;text-align:center;padding:4px'>{row['币种']}</div>",
                        unsafe_allow_html=True)

    macd_cols = st.columns([1.2] + [1] * len(rows))
    with macd_cols[0]:
        st.markdown("<div style='font-size:11px;color:#888;padding:6px 4px'>MACD</div>",
                    unsafe_allow_html=True)
    for i, row in enumerate(rows):
        with macd_cols[i + 1]:
            st.markdown(f"<div class='matrix-cell' style='background:{row['macd_color']}22;color:{row['macd_color']}'>{row['MACD状态']}</div>",
                        unsafe_allow_html=True)

    rsi_cols = st.columns([1.2] + [1] * len(rows))
    with rsi_cols[0]:
        st.markdown("<div style='font-size:11px;color:#888;padding:6px 4px'>RSI</div>",
                    unsafe_allow_html=True)
    for i, row in enumerate(rows):
        with rsi_cols[i + 1]:
            st.markdown(f"<div class='matrix-cell' style='background:{row['rsi_color']}22;color:{row['rsi_color']}'>{row['RSI']}</div>",
                        unsafe_allow_html=True)

    if mirror_any:
        st.markdown('<div class="hint-box" style="margin-top:10px">🪞 矩阵已切换为镜像视角：MACD方向翻转，RSI = 100 - 原值。绿色代表镜像后"多头"区域（即正常视角空头）。</div>',
                    unsafe_allow_html=True)
else:
    st.info("在左侧选择要对比的币种")

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='color:#334;font-size:11px;font-family:JetBrains Mono,monospace;text-align:center;padding:8px;'>
数据来源：OKX 公开接口 · 仅供学习训练，不构成投资建议
</div>
""", unsafe_allow_html=True)
