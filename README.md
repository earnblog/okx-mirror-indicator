# OKX Mirror Indicator 🪞

训练多空思维的镜像指标工具，接入 OKX 实时数据。

## 功能

- K线图 + EMA 20 / 52 / 200
- MACD 镜像翻转（以0轴为中心）
- RSI 镜像翻转（以50为中心，100-RSI）
- 思维提示：镜像后自动生成反向解读文字
- 多币种周期矩阵，一眼扫多个币种的MACD/RSI状态
- 支持全部 OKX 永续合约，交易对搜索切换
- 自动刷新（10秒）

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. 上传 `app.py` + `requirements.txt` 到 GitHub 仓库
2. 访问 share.streamlit.io
3. New app → 选仓库 → Main file: `app.py` → Deploy
