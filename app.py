import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

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
.mirror-badge { display:inline-block;background:#1a2a3a;color:#378ADD;border:1px solid #378ADD55;border-radius:6px;padding:3px 12px;font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace;margin-left:10px; }
.normal-badge { display:inline-block;background:#1a2a1a;color:#1D9E75;border:1px solid #1D9E7555;border-radius:6px;padding:3px 12px;font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace;margin-left:10px; }
.hint-box { background:#0d1a2a;border:1px solid #1a3a5a;border-left:3px solid #378ADD;border-radius:6px;padding:10px 14px;margin:8px 0;font-size:12px;color:#89b4e0;line-height:1.7; }
.hint-box-normal { background:#0d1a14;border:1px solid #1a3a2a;border-left:3px solid #1D9E75;border-radius:6px;padding:10px 14px;margin:8px 0;font-size:12px;color:#7ecfaa;line-height:1.7; }
.hint-box-warn { background:#1a1500;border:1px solid #3a3000;border-left:3px solid #EF9F27;border-radius:6px;padding:10px 14px;margin:8px 0;font-size:12px;color:#d4a843;line-height:1.7; }
.hint-box-danger { background:#1a0d0d;border:1px solid #3a1a1a;border-left:3px solid #D85A30;border-radius:6px;padding:10px 14px;margin:8px 0;font-size:12px;color:#e08060;line-height:1.7; }
.stat-card { background:#111318;border:1px solid #1e2230;border-radius:8px;padding:10px 14px;text-align:center; }
.stat-val { font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace; }
.stat-label { font-size:11px;color:#666;margin-top:2px; }
.ctx-row { display:flex;align-items:center;gap:10px;padding:7px 12px;border-radius:6px;margin-bottom:5px;background:#111318;border:1px solid #1e2230;font-family:'JetBrains Mono',monospace;font-size:12px; }
.ctx-tf { color:#555;width:36px;font-weight:700; }
.ctx-badge { display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;min-width:110px;text-align:center; }
.analysis-box { background:#0d0f14;border:1px solid #1e2230;border-radius:8px;padding:14px 18px;margin:10px 0;line-height:1.9;font-size:13px;color:#ccc; }
.analysis-box b { color:#eee; }
</style>
""", unsafe_allow_html=True)

BASE = "https://www.okx.com/api/v5"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
POPULAR = ["BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","BNB-USDT-SWAP","XRP-USDT-SWAP",
           "DOGE-USDT-SWAP","ADA-USDT-SWAP","AVAX-USDT-SWAP","LINK-USDT-SWAP","DOT-USDT-SWAP"]
TIMEFRAMES = {"1m":"1m","5m":"5m","15m":"15m","1h":"1H","4h":"4H","1d":"1D"}
PARENT_TF = {"1m":["5m","15m","1h"],"5m":["15m","1h","4h"],
             "15m":["1h","4h","1d"],"1h":["4h","1d"],"4h":["1d"],"1d":[]}

@st.cache_data(ttl=10)
def fetch_candles(inst_id, bar, limit=200):
    try:
        r = requests.get(f"{BASE}/market/candles",
                         params={"instId":inst_id,"bar":bar,"limit":limit},
                         headers=HEADERS, timeout=8)
        data = r.json()
        if data.get("code") != "0":
            return pd.DataFrame()
        df = pd.DataFrame(data["data"],
                          columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df = df.astype({"ts":float,"open":float,"high":float,"low":float,"close":float,"vol":float})
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

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, sig=9):
    diff = calc_ema(close, fast) - calc_ema(close, slow)
    dea = calc_ema(diff, sig)
    return diff, dea, (diff - dea) * 2

def calc_rsi(close, p=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def calc_support_resistance(df, lookback=80):
    if len(df) < 10:
        return [], []
    highs = df["high"].iloc[-lookback:]
    lows = df["low"].iloc[-lookback:]
    close_last = df["close"].iloc[-1]
    price_range = highs.max() - lows.min()
    if price_range == 0:
        return [], []
    tolerance = price_range * 0.015
    h = df["high"].iloc[-lookback:].values
    l = df["low"].iloc[-lookback:].values
    swing_highs, swing_lows = [], []
    for i in range(2, len(h) - 2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append(h[i])
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append(l[i])
    ema20_last = calc_ema(df["close"], 20).iloc[-1]
    ema52_last = calc_ema(df["close"], 52).iloc[-1]
    ema200_last = calc_ema(df["close"], 200).iloc[-1]
    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clusters, group = [], [levels[0]]
        for v in levels[1:]:
            if v - group[-1] < tolerance:
                group.append(v)
            else:
                clusters.append(float(np.mean(group)))
                group = [v]
        clusters.append(float(np.mean(group)))
        return clusters
    sup_levels = cluster(swing_lows)
    res_levels = cluster(swing_highs)
    for e in [ema20_last, ema52_last, ema200_last]:
        if e < close_last:
            sup_levels.append(e)
        else:
            res_levels.append(e)
    supports = sorted([s for s in sup_levels if s < close_last], reverse=True)[:3]
    resistances = sorted([r for r in res_levels if r > close_last])[:3]
    return supports, resistances

def get_macd_state(hist_val, diff_val):
    if diff_val > 0 and hist_val > 0:
        return "0轴上方·扩张", "#1D9E75"
    elif diff_val > 0 and hist_val <= 0:
        return "0轴上方·收缩", "#EF9F27"
    elif diff_val <= 0 and hist_val < 0:
        return "0轴下方·扩张", "#D85A30"
    else:
        return "0轴下方·收缩", "#EF9F27"

def get_hist_momentum(hist_s, mirror=False):
    h = hist_s.dropna().copy()
    if mirror:
        h = -h
    if len(h) < 4:
        return "数据不足", "#555", "unknown"
    last, prev, prev2 = h.iloc[-1], h.iloc[-2], h.iloc[-3]
    if prev <= 0 < last:
        return "↑ 上穿0轴", "#1D9E75", "cross_up"
    if prev >= 0 > last:
        return "↓ 下穿0轴", "#D85A30", "cross_down"
    avg_abs = h.abs().iloc[-10:].mean()
    if avg_abs > 0 and abs(last) < avg_abs * 0.15:
        tag = "near_zero_up" if last > 0 else "near_zero_down"
        return ("→ 逼近0轴↑" if last > 0 else "→ 逼近0轴↓"), "#EF9F27", tag
    if last > 0:
        if abs(last) > abs(prev) and abs(prev) > abs(prev2):
            return "▲ 上方扩张", "#1D9E75", "expand_up"
        elif abs(last) < abs(prev):
            return "▼ 上方收缩", "#EF9F27", "shrink_up"
        else:
            return "─ 上方平稳", "#5DCAA5", "flat_up"
    else:
        if abs(last) > abs(prev) and abs(prev) > abs(prev2):
            return "▼ 下方扩张", "#D85A30", "expand_down"
        elif abs(last) < abs(prev):
            return "▲ 下方收缩", "#EF9F27", "shrink_down"
        else:
            return "─ 下方平稳", "#F0997B", "flat_down"

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

def get_tf_data(inst_id, tf_key, mirror=False):
    bar = TIMEFRAMES[tf_key]
    df = fetch_candles(inst_id, bar, 200)
    if df.empty or len(df) < 30:
        return None
    close = df["close"]
    diff_s, dea_s, hist_s = calc_macd(close)
    rsi_s = calc_rsi(close)
    last_diff = diff_s.iloc[-1]
    last_dea = dea_s.iloc[-1]
    last_hist = hist_s.iloc[-1]
    last_rsi = rsi_s.iloc[-1]
    raw_rsi = float(last_rsi)
    if mirror:
        last_diff = -last_diff
        last_dea = -last_dea
        last_hist = -last_hist
        last_rsi = 100 - last_rsi
    macd_txt, macd_col = get_macd_state(last_hist, last_diff)
    hist_txt, hist_col, hist_tag = get_hist_momentum(hist_s, mirror)
    rsi_txt, rsi_col = get_rsi_state(last_rsi)
    ema200 = calc_ema(close, 200)
    trend_up = float(close.iloc[-1]) > float(ema200.iloc[-1])
    sup, res = calc_support_resistance(df)
    return {
        "tf": tf_key, "df": df,
        "diff": diff_s, "dea": dea_s, "hist": hist_s, "rsi": rsi_s,
        "ema20": calc_ema(close, 20), "ema52": calc_ema(close, 52), "ema200": ema200,
        "last_diff": last_diff, "last_dea": last_dea, "last_hist": last_hist,
        "last_rsi": last_rsi, "raw_rsi": raw_rsi,
        "macd_txt": macd_txt, "macd_col": macd_col,
        "hist_txt": hist_txt, "hist_col": hist_col, "hist_tag": hist_tag,
        "rsi_txt": rsi_txt, "rsi_col": rsi_col,
        "trend_up": trend_up, "close": float(close.iloc[-1]),
        "supports": sup, "resistances": res,
    }

def build_analysis(main_data, parent_data_list, mirror):
    if not main_data:
        return "", "hint-box"
    mtf = main_data["tf"]
    mh_tag = main_data["hist_tag"]
    mrsi = main_data["raw_rsi"]
    lines = []
    box_class = "hint-box-normal"

    lines.append(f"<b>【{mtf} 主周期】</b> MACD {main_data['macd_txt']}，柱线 {main_data['hist_txt']}，RSI {main_data['rsi_txt']}")

    bearish_count = 0
    bullish_count = 0
    exhausted_count = 0

    if parent_data_list:
        lines.append("")
        lines.append("<b>【上级周期压制】</b>")
        for pd_ in parent_data_list:
            if pd_ is None:
                continue
            ptf = pd_["tf"]
            ph_tag = pd_["hist_tag"]
            pm_txt = pd_["macd_txt"]

            if "expand_down" in ph_tag or "cross_down" in ph_tag:
                desc = f"<span style='color:#D85A30'>{ptf}：{pm_txt} / {pd_['hist_txt']} → 空头压制强，下级反弹空间有限</span>"
                bearish_count += 1
            elif "expand_up" in ph_tag or "cross_up" in ph_tag:
                desc = f"<span style='color:#D85A30'>{ptf}：{pm_txt} / {pd_['hist_txt']} → 多头仍强，下级下跌是回调非反转</span>"
                bearish_count += 1
            elif "shrink_down" in ph_tag:
                desc = f"<span style='color:#EF9F27'>{ptf}：{pm_txt} / {pd_['hist_txt']} → 空头衰减中，下级反弹空间扩大</span>"
                exhausted_count += 1
            elif "shrink_up" in ph_tag:
                desc = f"<span style='color:#EF9F27'>{ptf}：{pm_txt} / {pd_['hist_txt']} → 多头减弱，对下级支撑减少</span>"
                exhausted_count += 1
            elif "near_zero" in ph_tag:
                desc = f"<span style='color:#EF9F27'>{ptf}：{pm_txt} / {pd_['hist_txt']} ⚡ 归零轴中，方向未明</span>"
                exhausted_count += 1
            else:
                desc = f"<span style='color:#888'>{ptf}：{pm_txt} / {pd_['hist_txt']}</span>"
                bullish_count += 1

            lines.append(f"&nbsp;&nbsp;· {desc}")

    lines.append("")
    lines.append("<b>【综合判断】</b>")
    at_zero = "near_zero" in mh_tag or "cross" in mh_tag

    if at_zero:
        if bearish_count >= 2:
            conclusion = ("⚠️ 主周期柱线正在归零轴，上级周期多数偏空压制。"
                "此处反弹概率低、空间小，大概率短暂触碰0轴后继续下穿。"
                "<b style='color:#D85A30'> 建议：不在0轴附近做多，等下穿0轴后柱线扩张再顺势做空。</b>")
            box_class = "hint-box-danger"
        elif exhausted_count >= 2:
            conclusion = ("⚡ 主周期柱线归零轴，上级周期空头动能多数在衰减。"
                "反弹有一定空间，但上级未确认反转前仍需谨慎。"
                "<b style='color:#EF9F27'> 建议：可小仓试多，但设紧止损，等待上级周期也出现收缩转向信号。</b>")
            box_class = "hint-box-warn"
        elif bullish_count >= 2:
            conclusion = ("✅ 主周期柱线归零轴，上级周期多头支撑较强。"
                "此处反弹概率较高。"
                "<b style='color:#1D9E75'> 建议：等待价格企稳信号（锤子线/放量、RSI超卖回头）后做多。</b>")
            box_class = "hint-box-normal"
        else:
            conclusion = ("→ 主周期柱线归零轴，上级周期方向混合，多空博弈区间。不宜重仓，等待方向明确再行动。")
            box_class = "hint-box-warn"
    elif "expand_down" in mh_tag:
        if bearish_count >= 2:
            conclusion = ("📉 主周期空头扩张 + 上级周期同步偏空，多周期共振向下。"
                "<b style='color:#D85A30'> 趋势明确，顺势做空，反弹是机会不是威胁。</b>")
            box_class = "hint-box-danger"
        else:
            conclusion = "📉 主周期空头扩张，但上级周期有支撑，下跌幅度可能有限，注意反转信号。"
            box_class = "hint-box-warn"
    elif "expand_up" in mh_tag:
        if bullish_count >= 2:
            conclusion = ("📈 主周期多头扩张 + 上级周期同步偏多，多周期共振向上。"
                "<b style='color:#1D9E75'> 趋势明确，顺势做多，回调是机会不是威胁。</b>")
            box_class = "hint-box-normal"
        elif bearish_count >= 2:
            conclusion = ("⚠️ 主周期多头扩张，但上级周期偏空压制。"
                "当前上涨可能是反弹而非反转。<b style='color:#EF9F27'> 不追高，等上级周期也转多后再参与。</b>")
            box_class = "hint-box-warn"
        else:
            conclusion = "📈 主周期多头扩张，上级周期方向混合，当前上涨有一定空间但需持续观察。"
            box_class = "hint-box-warn"
    elif "shrink" in mh_tag:
        direction = "多头" if "up" in mh_tag else "空头"
        conclusion = (f"→ 主周期{direction}动能收缩，趋势在减弱。"
            "等待柱线方向明确后再决策，不宜追单。")
        box_class = "hint-box-warn"
    else:
        conclusion = "→ 指标平稳，无明显信号，观望为主。"
        box_class = "hint-box"

    lines.append(conclusion)

    sup = main_data["supports"]
    res = main_data["resistances"]
    close_p = main_data["close"]
    if sup or res:
        lines.append("")
        lines.append("<b>【关键价位】</b>")
        if res:
            res_str = " / ".join([f"{r:.5g}" for r in res])
            lines.append(f"&nbsp;&nbsp;· 上方阻力：<span style='color:#D85A30'>{res_str}</span>")
        if sup:
            sup_str = " / ".join([f"{s:.5g}" for s in sup])
            sup_dist = (close_p - sup[0]) / close_p * 100
            lines.append(f"&nbsp;&nbsp;· 下方支撑：<span style='color:#1D9E75'>{sup_str}</span>（最近支撑距当前 {sup_dist:.1f}%）")
            if sup_dist < 1.5 and at_zero:
                lines.append("&nbsp;&nbsp;· <span style='color:#EF9F27'>⚡ 价格接近支撑 + MACD归零轴同时出现，是关键博弈点。支撑守住则反弹，支撑失守则加速下跌。</span>")

    if mrsi <= 32:
        lines.append(f"&nbsp;&nbsp;· RSI超卖区（{mrsi:.1f}），配合支撑位可作为反弹参考信号，等K线企稳确认")
    elif mrsi >= 68:
        lines.append(f"&nbsp;&nbsp;· RSI超买区（{mrsi:.1f}），追多风险增大，注意回调压力")

    if mirror:
        lines.append("")
        lines.append("<b>【🪞 镜像在问你】</b>")
        if "down" in mh_tag:
            lines.append("镜像视角是空头，对应现实是多头。问自己：如果这是多头，你敢不敢追？用同样的标准审视你的空单。")
        elif "up" in mh_tag:
            lines.append("镜像视角是多头，对应现实是空头。问自己：如果这是多头你会担心追高吗？那现在的空头，是否也存在同样的追空风险？")
        else:
            lines.append("镜像中性区间。问自己：你的方向判断除了指标，还有什么依据？")

    return "<br>".join(lines), box_class

def draw_kline(data, display_n=100):
    import plotly.graph_objects as go
    df = data["df"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="K线",
        increasing_fillcolor="#1D9E75", increasing_line_color="#1D9E75",
        decreasing_fillcolor="#D85A30", decreasing_line_color="#D85A30",
    ))
    for val, name, color in [(data["ema20"],"EMA20","#E24B4A"),(data["ema52"],"EMA52","#EF9F27"),(data["ema200"],"EMA200","#1D9E75")]:
        fig.add_trace(go.Scatter(x=df["ts"], y=val, name=name, line=dict(color=color, width=1.2)))
    for s in data["supports"]:
        fig.add_hline(y=s, line_color="#1D9E75", line_width=0.8, line_dash="dot",
                      annotation_text=f"S {s:.5g}", annotation_position="right",
                      annotation_font_color="#1D9E75", annotation_font_size=10)
    for r in data["resistances"]:
        fig.add_hline(y=r, line_color="#D85A30", line_width=0.8, line_dash="dot",
                      annotation_text=f"R {r:.5g}", annotation_position="right",
                      annotation_font_color="#D85A30", annotation_font_size=10)
    ts = df["ts"]
    x_end = ts.iloc[-1]
    x_start = ts.iloc[max(0, len(ts) - display_n)]
    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0d0f14",
        font=dict(color="#aaa", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1e2230", rangeslider_visible=False,
                   range=[x_start, x_end], fixedrange=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2230", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=90, t=30, b=10), height=300,
    )
    return fig

def draw_macd(data, mirror, display_n=100):
    import plotly.graph_objects as go
    df = data["df"]
    d = (-data["diff"]) if mirror else data["diff"]
    de = (-data["dea"]) if mirror else data["dea"]
    h = (-data["hist"]) if mirror else data["hist"]
    colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in h.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["ts"], y=h, name="柱", marker_color=colors, marker_line_width=0, opacity=0.85))
    fig.add_trace(go.Scatter(x=df["ts"], y=d, name="DIFF", line=dict(color="#378ADD", width=1.4)))
    fig.add_trace(go.Scatter(x=df["ts"], y=de, name="DEA", line=dict(color="#E24B4A", width=1.4)))
    fig.add_hline(y=0, line_color="#334", line_width=1, line_dash="dot")
    if mirror:
        fig.add_annotation(x=0.01, y=0.95, xref="paper", yref="paper", text="🪞 MACD 镜像",
                           showarrow=False, font=dict(color="#378ADD", size=11),
                           bgcolor="#0d1a2a", bordercolor="#378ADD", borderwidth=1, borderpad=4)
    ts = df["ts"]
    x_end = ts.iloc[-1]
    x_start = ts.iloc[max(0, len(ts) - display_n)]
    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0d0f14",
        font=dict(color="#aaa", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1e2230",
                   range=[x_start, x_end], fixedrange=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2230", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=90, t=10, b=10), height=340, barmode="relative",
    )
    return fig

def draw_rsi(data, mirror, display_n=100):
    import plotly.graph_objects as go
    df = data["df"]
    r = (100 - data["rsi"]) if mirror else data["rsi"]
    fig = go.Figure()
    fig.add_hline(y=70, line_color="#D85A30", line_width=0.8, line_dash="dot",
                  annotation_text="70", annotation_position="right")
    fig.add_hline(y=50, line_color="#444", line_width=0.8, line_dash="dot")
    fig.add_hline(y=30, line_color="#1D9E75", line_width=0.8, line_dash="dot",
                  annotation_text="30", annotation_position="right")
    fig.add_trace(go.Scatter(x=df["ts"], y=r, name="RSI",
                             line=dict(color="#7F77DD", width=1.5),
                             fill="tozeroy", fillcolor="rgba(127,119,221,0.08)"))
    if mirror:
        fig.add_annotation(x=0.01, y=0.95, xref="paper", yref="paper", text="🪞 RSI 镜像",
                           showarrow=False, font=dict(color="#378ADD", size=11),
                           bgcolor="#0d1a2a", bordercolor="#378ADD", borderwidth=1, borderpad=4)
    ts = df["ts"]
    x_end = ts.iloc[-1]
    x_start = ts.iloc[max(0, len(ts) - display_n)]
    fig.update_layout(
        paper_bgcolor="#0a0c10", plot_bgcolor="#0d0f14",
        font=dict(color="#aaa", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1e2230",
                   range=[x_start, x_end], fixedrange=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2230", side="right", range=[-5, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=90, t=10, b=10), height=130,
    )
    return fig

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
    tf = st.selectbox("主周期", list(TIMEFRAMES.keys()), index=2)
    st.markdown("---")
    st.markdown("### 🪞 镜像控制")
    macd_mirror = st.toggle("MACD 镜像翻转", value=False)
    rsi_mirror = st.toggle("RSI 镜像翻转", value=False)
    st.markdown("---")
    display_n = st.slider("显示K线根数", min_value=50, max_value=200, value=100, step=10)
    st.markdown("---")
    auto_refresh = st.toggle("自动刷新 (15秒)", value=False)
    if st.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()

if auto_refresh:
    time.sleep(15)
    st.cache_data.clear()
    st.rerun()

# ── Load ──────────────────────────────────────────────────────────────────────
mirror_any = macd_mirror or rsi_mirror
mode_badge = '<span class="mirror-badge">🪞 镜像模式</span>' if mirror_any else '<span class="normal-badge">✓ 正常模式</span>'
st.markdown(f"## {selected_symbol} · {tf} {mode_badge}", unsafe_allow_html=True)

with st.spinner("加载数据..."):
    main_data = get_tf_data(selected_symbol, tf, mirror_any)
    parent_tfs = PARENT_TF.get(tf, [])
    parent_data = [get_tf_data(selected_symbol, ptf, mirror_any) for ptf in parent_tfs]

if not main_data:
    st.error("数据加载失败，请稍后重试")
    st.stop()

# ── Stats ─────────────────────────────────────────────────────────────────────
close_p = main_data["close"]
prev_close = float(main_data["df"]["close"].iloc[-2])
pct_chg = (close_p - prev_close) / prev_close * 100
chg_color = "#1D9E75" if pct_chg >= 0 else "#D85A30"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{chg_color}">{close_p:,.5g}</div><div class="stat-label">价格 ({pct_chg:+.2f}%)</div></div>', unsafe_allow_html=True)
with c2:
    d_disp = main_data["last_diff"]
    dc = "#1D9E75" if d_disp >= 0 else "#D85A30"
    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{dc}">{d_disp:.4f}</div><div class="stat-label">DIFF{"(镜)" if macd_mirror else ""}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{main_data["hist_col"]};font-size:14px">{main_data["hist_txt"]}</div><div class="stat-label">柱线状态</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{main_data["rsi_col"]};font-size:15px">{main_data["rsi_txt"]}</div><div class="stat-label">RSI{"(镜)" if rsi_mirror else ""}</div></div>', unsafe_allow_html=True)
with c5:
    if mirror_any:
        trend_txt = "空头趋势(镜)" if main_data["trend_up"] else "多头趋势(镜)"
        trend_col = "#D85A30" if main_data["trend_up"] else "#1D9E75"
    else:
        trend_txt = "多头趋势" if main_data["trend_up"] else "空头趋势"
        trend_col = "#1D9E75" if main_data["trend_up"] else "#D85A30"
    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{trend_col};font-size:14px">{trend_txt}</div><div class="stat-label">EMA200</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Upper timeframe bar ───────────────────────────────────────────────────────
if parent_data:
    st.markdown("**上级周期状态**")
    ctx_html = ""
    for pd_ in parent_data:
        if pd_ is None:
            continue
        ph_tag = pd_["hist_tag"]
        if "expand_down" in ph_tag or "cross_down" in ph_tag:
            arrow = "⬇️ 空头压制"
        elif "expand_up" in ph_tag or "cross_up" in ph_tag:
            arrow = "⬆️ 多头支撑"
        elif "near_zero" in ph_tag:
            arrow = "⚡ 归零轴"
        elif "shrink" in ph_tag:
            arrow = "↔️ 动能衰减"
        else:
            arrow = "─"
        ctx_html += f"""<div class="ctx-row">
            <span class="ctx-tf">{pd_['tf']}</span>
            <span class="ctx-badge" style="background:{pd_['macd_col']}22;color:{pd_['macd_col']}">{pd_['macd_txt']}</span>
            <span class="ctx-badge" style="background:{pd_['hist_col']}22;color:{pd_['hist_col']}">{pd_['hist_txt']}</span>
            <span class="ctx-badge" style="background:{pd_['rsi_col']}22;color:{pd_['rsi_col']}">{pd_['rsi_txt']}</span>
            <span style="color:#666;font-size:11px">{arrow}</span>
        </div>"""
    st.markdown(ctx_html, unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
analysis_txt, box_cls = build_analysis(main_data, parent_data, mirror_any)
st.markdown(f'<div class="analysis-box">{analysis_txt}</div>', unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("**K线图** · EMA 20 / 52 / 200 · 支撑阻力（不参与镜像）")
st.plotly_chart(draw_kline(main_data, display_n), use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})

st.markdown(f"**MACD {'🪞 镜像' if macd_mirror else '正常'}**")
st.plotly_chart(draw_macd(main_data, macd_mirror, display_n), use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})



st.markdown("---")
st.markdown("<div style='color:#334;font-size:11px;font-family:JetBrains Mono,monospace;text-align:center;padding:8px;'>数据来源：OKX 公开接口 · 仅供学习训练，不构成投资建议</div>", unsafe_allow_html=True)
