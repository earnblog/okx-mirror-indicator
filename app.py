import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="OKX MACD Mirror",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0c10; color: #e0e0e0; }
[data-testid="stSidebar"] { background: #0d0f14 !important; border-right: 1px solid #1e2230; }
[data-testid="stSidebar"] * { color: #ccc !important; }
h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; color: #e0e0e0 !important; }
.state-badge {
    display:inline-block; padding:2px 10px; border-radius:4px;
    font-size:11px; font-weight:600; font-family:'JetBrains Mono',monospace;
}
.mirror-on {
    display:inline-block; background:#1a2a3a; color:#378ADD;
    border:1px solid #378ADD88; border-radius:4px; padding:1px 8px; font-size:11px;
}
.divider { height:2px; background:#1a1e2a; margin:4px 0 10px; }
</style>
""", unsafe_allow_html=True)

BASE = "https://www.okx.com/api/v5"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

POPULAR = ["BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","BNB-USDT-SWAP",
           "XRP-USDT-SWAP","DOGE-USDT-SWAP","ADA-USDT-SWAP","AVAX-USDT-SWAP",
           "LINK-USDT-SWAP","DOT-USDT-SWAP"]

TIMEFRAMES = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1H","2h":"2H","4h":"4H","6h":"6H","12h":"12H",
    "1d":"1D","2d":"2D","1w":"1W","1M":"1M"
}

TF_LABELS = {
    "1m":"1分钟","3m":"3分钟","5m":"5分钟","15m":"15分钟","30m":"30分钟",
    "1h":"1小时","2h":"2小时","4h":"4小时","6h":"6小时","12h":"12小时",
    "1d":"日线","2d":"2日","1w":"周线","1M":"月线"
}

TF_KEYS = list(TIMEFRAMES.keys())
DEFAULT_TFS = ["1m", "5m", "15m", "1h"]

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=8)
def fetch_candles(inst_id, bar, limit=300):
    try:
        r = requests.get(f"{BASE}/market/candles",
                         params={"instId": inst_id, "bar": bar, "limit": limit},
                         headers=HEADERS, timeout=8)
        data = r.json()
        if data.get("code") != "0":
            return pd.DataFrame()
        df = pd.DataFrame(data["data"],
                          columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df = df.astype({"ts":float,"open":float,"high":float,
                        "low":float,"close":float,"vol":float})
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df.sort_values("ts").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_instruments():
    try:
        r = requests.get(f"{BASE}/public/instruments",
                         params={"instType":"SWAP"}, headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("code") != "0":
            return []
        return [d["instId"] for d in data["data"] if d["instId"].endswith("USDT-SWAP")]
    except Exception:
        return []

# ── Indicators ────────────────────────────────────────────────────────────────

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, sig=9):
    diff = calc_ema(close, fast) - calc_ema(close, slow)
    dea  = calc_ema(diff, sig)
    hist = (diff - dea) * 2
    return diff, dea, hist

def get_state(diff_val, hist_val):
    if diff_val > 0 and hist_val > 0:
        return "0轴上方·扩张", "#1D9E75"
    elif diff_val > 0 and hist_val <= 0:
        return "0轴上方·收缩", "#EF9F27"
    elif diff_val <= 0 and hist_val < 0:
        return "0轴下方·扩张", "#D85A30"
    else:
        return "0轴下方·收缩", "#EF9F27"

def get_momentum(hist_s, mirror=False):
    h = hist_s.dropna().copy()
    if mirror:
        h = -h
    if len(h) < 4:
        return "数据不足", "#555"
    last, prev, prev2 = h.iloc[-1], h.iloc[-2], h.iloc[-3]
    if prev <= 0 < last:
        return "↑ 上穿0轴 ⚡", "#1D9E75"
    if prev >= 0 > last:
        return "↓ 下穿0轴 ⚡", "#D85A30"
    avg_abs = h.abs().iloc[-10:].mean()
    if avg_abs > 0 and abs(last) < avg_abs * 0.15:
        return ("→ 逼近0轴↑ ⚡" if last >= 0 else "→ 逼近0轴↓ ⚡"), "#EF9F27"
    if last > 0:
        if abs(last) > abs(prev) and abs(prev) > abs(prev2):
            return "▲ 上方扩张", "#1D9E75"
        elif abs(last) < abs(prev):
            return "▼ 上方收缩", "#EF9F27"
        else:
            return "─ 上方平稳", "#5DCAA5"
    else:
        if abs(last) > abs(prev) and abs(prev) > abs(prev2):
            return "▼ 下方扩张", "#D85A30"
        elif abs(last) < abs(prev):
            return "▲ 下方收缩", "#EF9F27"
        else:
            return "─ 下方平稳", "#F0997B"

# ── Draw ──────────────────────────────────────────────────────────────────────

def draw_macd(df, diff, dea, hist, mirror, display_n, height=260):
    import plotly.graph_objects as go

    d  = (-diff) if mirror else diff
    de = (-dea)  if mirror else dea
    h  = (-hist) if mirror else hist

    colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in h.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["ts"], y=h, name="柱",
                         marker_color=colors, marker_line_width=0, opacity=0.88))
    fig.add_trace(go.Scatter(x=df["ts"], y=d,  name="DIFF",
                             line=dict(color="#378ADD", width=1.6)))
    fig.add_trace(go.Scatter(x=df["ts"], y=de, name="DEA",
                             line=dict(color="#E24B4A", width=1.6)))
    fig.add_hline(y=0, line_color="#2a2e3a", line_width=1.2, line_dash="dot")

    if mirror:
        fig.add_annotation(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="🪞 镜像", showarrow=False,
            font=dict(color="#378ADD", size=11),
            bgcolor="#0d1a2a", bordercolor="rgba(55,138,221,0.27)",
            borderwidth=1, borderpad=4,
        )

    ts = df["ts"]
    x_start = ts.iloc[max(0, len(ts) - display_n)]
    x_end   = ts.iloc[-1]

    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0d0f14",
        font=dict(color="#aaa", size=10),
        xaxis=dict(showgrid=True, gridcolor="#1a1e2a",
                   range=[x_start, x_end], fixedrange=False,
                   tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#1a1e2a",
                   side="right", tickfont=dict(size=9)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=4, r=60, t=6, b=6),
        height=height,
        barmode="relative",
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🪞 MACD Mirror")
    st.markdown("---")

    all_instruments = fetch_instruments()
    search_input = st.text_input("搜索交易对", placeholder="BTC, ETH, SOL...")
    if search_input:
        filtered = [i for i in all_instruments if search_input.upper() in i][:20]
        options = filtered if filtered else POPULAR
    else:
        options = POPULAR

    symbol = st.selectbox("交易对", options, index=0)

    st.markdown("---")
    st.markdown("### 显示设置")
    display_n = st.slider("显示K线根数", min_value=50, max_value=300, value=100, step=10)

    st.markdown("---")
    auto_refresh = st.toggle("自动刷新 (15秒)", value=False)
    if st.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()

if auto_refresh:
    time.sleep(15)
    st.cache_data.clear()
    st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

sym_label = symbol.replace("-USDT-SWAP", "").replace("-USDT", "")
st.markdown(f"## 🪞 {sym_label} · MACD Mirror")
st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

# ── 4 MACD Panels ─────────────────────────────────────────────────────────────

for i in range(4):
    c_tf, c_label, c_mirror = st.columns([1.4, 2.6, 1])

    with c_tf:
        default_idx = TF_KEYS.index(DEFAULT_TFS[i])
        tf_key = st.selectbox(
            f"panel_{i}",
            options=TF_KEYS,
            index=default_idx,
            format_func=lambda k: TF_LABELS[k],
            key=f"tf_{i}",
            label_visibility="collapsed",
        )

    with c_mirror:
        mirror = st.toggle("🪞", key=f"mirror_{i}", value=False)

    # Fetch data
    bar = TIMEFRAMES[tf_key]
    df  = fetch_candles(symbol, bar, 300)

    if df.empty or len(df) < 30:
        with c_label:
            st.markdown(f"<div style='padding:8px 0;color:#555;font-size:12px'>⚠️ {TF_LABELS[tf_key]} 数据加载失败</div>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        continue

    close = df["close"]
    diff_s, dea_s, hist_s = calc_macd(close)

    d_val = float(-diff_s.iloc[-1] if mirror else diff_s.iloc[-1])
    h_val = float(-hist_s.iloc[-1] if mirror else hist_s.iloc[-1])

    state_txt, state_col = get_state(d_val, h_val)
    mom_txt,   mom_col   = get_momentum(hist_s, mirror)

    with c_label:
        mirror_tag = "<span class='mirror-on'>🪞 镜像</span>&nbsp;&nbsp;" if mirror else ""
        badge_html = (
            f"<div style='padding:6px 0 0;line-height:2'>"
            f"{mirror_tag}"
            f"<span class='state-badge' style='background:{state_col}22;color:{state_col}'>{state_txt}</span>"
            f"&nbsp;&nbsp;"
            f"<span class='state-badge' style='background:{mom_col}22;color:{mom_col}'>{mom_txt}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='font-size:10px;color:#444;font-family:JetBrains Mono,monospace'>DIFF {d_val:+.4f}</span>"
            f"</div>"
        )
        st.markdown(badge_html, unsafe_allow_html=True)

    fig = draw_macd(df, diff_s, dea_s, hist_s, mirror, display_n, height=260)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": True})

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='color:#1e2230;font-size:11px;font-family:JetBrains Mono,monospace;"
    "text-align:center;padding:10px;'>OKX 公开接口 · 仅供学习训练，不构成投资建议</div>",
    unsafe_allow_html=True,
)
