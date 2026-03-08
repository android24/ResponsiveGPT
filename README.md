# ResponsiveGPT Integrated Project (highD + roundD)

一个轻量、可执行、低耦合高内聚的工程版本，集成：

- Ollama 本地 embedding
- jiekou GPT-5.2（OpenAI 兼容接口）
- RAG 规则检索
- 在线驾驶人画像更新
- runs/ 实验日志记录
- 评估模块（TTC / 违规率）
- highD 强交互事件接入
- roundD 高风险事件接入

## 目录说明

- `src/responsivegpt/domain/`：领域模型、抽象接口、纯业务逻辑
- `src/responsivegpt/application/`：主编排服务
- `src/responsivegpt/infrastructure/`：LLM / embedding / 向量库 / profile repo / 规则库
- `src/responsivegpt/interface/adapters/`：highD / roundD 数据适配器
- `src/responsivegpt/interface/`：CLI 与批处理入口
- `src/responsivegpt/evaluation/`：指标与 run 日志

## 安装

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
```

## 配置

复制 `.env.example` 为 `.env`，填写你的 key：

```bash
cp .env.example .env
```

## 运行 demo

```bash
python -m src.responsivegpt.interface.cli --demo --tag demo
```

## 跑 highD 批处理

```bash
python -m src.responsivegpt.interface.run_highd_batch \
  --csv_path /path/to/highd_strong_interactions_full.csv \
  --driver_type 激进 \
  --feedback "保持效率，但避免明显危险操作" \
  --tag highd_run
```

## 跑 roundD 批处理

```bash
python -m src.responsivegpt.interface.run_roundd_batch \
  --csv_path /path/to/all_high_risk_events_summary.csv \
  --driver_type 保守 \
  --feedback "在环岛场景中优先安全，避免激进汇入与抢行" \
  --tag roundd_run
```

## 输出

每次运行会自动生成：

`runs/<timestamp>_<tag>/`

包含：

- `config.json`
- `decisions.jsonl`
- `metrics.csv`
- `summary.json`
