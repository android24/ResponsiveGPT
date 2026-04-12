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
  --csv_path highd_strong_interactions_full.csv \  
  --model_role primary \                               
  --tag highd_test \                               
  --profile_name aggressive \                          
  --limit 50
```

## 跑 roundD 批处理
```bash
python -m src.responsivegpt.interface.run_round_batch \    
  --csv_path roundD_high_risk_events_summary.csv \ 
  --model_role primary \                               
  --tag round_test \                               
  --profile_name conservative \                        
  --limit 50 
```

## 跑 inD 批处理
```bash
 python -m src.responsivegpt.interface.run_ind_batch \
  --csv_path output_ind_risk_v4/top_200_risk_events_v4.csv \
  --model_role primary \                               
  --tag ind_test \                               
  --profile_name
```

## 跑 highD episode处理
```bash
python -m src.responsivegpt.interface.run_highd_episode_batch \
  --csv_path highd_strong_interactions_full.csv \    
  --profile_name aggressive \
  --tag highd_aggressive \
  --limit 10
```

## 跑 roundD episode处理
```bash
python -m src.responsivegpt.interface.run_round_episode_batch \
  --summary_csv roundD_high_risk_events_summary.csv \
  --clips_root clips \
  --profile_name aggressive \
  --tag round_aggressive \
  --limit 10
```

## 跑 inD episode处理
```bash
python -m src.responsivegpt.interface.run_ind_episode_batch \                
  --summary_csv all_risk_events_v4.csv \        
  --scenes_root output_ind_risk_v4/scenes \            
  --tag ind_episode_mini \                       
  --profile_name conservative \
  --model_role cheap
  
python -m src.responsivegpt.interface.run_ind_episode_batch \                      
  --summary_csv all_risk_events_v4.csv \        
  --scenes_root output_ind_risk_v4/scenes \            
  --tag ind_episode_gpt41 \                      
  --profile_name conservative \
  --model_role fallback
```

## 跑对比试验模块
### 分别支持 batch 的对比
比如比较三个数据集的 batch：
```bash
python -m src.responsivegpt.interface.compare_runner \
  --mode batch \
  --items \
    highD=runs/highd_batch/summary.json \
    rounD=runs/round_batch/summary.json \
    inD=runs/ind_batch/summary.json \
  --output_dir runs/compare_batch_all
  
python -m src.responsivegpt.interface.compare_experiments \
  --mode batch \
  --items \
    highD=runs/20260412_144107_highd_test_aggressive/summary.json \
    rounD=runs/20260412_142942_round_test_conservative/summary.json \
    inD=runs/20260412_132721_ind_test_conservative/summary.json \
  --output_dir runs/compare_highd_episode_profiles  
```
也可以比较同一数据集不同 profile：
```bash
python -m src.responsivegpt.interface.compare_runner \
  --mode batch \
  --items \
    aggressive=runs/ind_batch_aggressive/summary.json \
    balanced=runs/ind_batch_balanced/summary.json \
    conservative=runs/ind_batch_conservative/summary.json \
  --output_dir runs/compare_ind_batch_profiles
```
画图：
```bash
python -m src.responsivegpt.interface.compare_plotter \
  --json_path runs/compare_batch_all/batch_compare.json \
  --output_dir runs/compare_batch_all/plots
```  
  
### 分别支持 episode 的对比
比较三个数据集的 episode：
```bash
python -m src.responsivegpt.interface.compare_runner \
  --mode episode \
  --items \
    highD=runs/highd_episode/summary.json \
    rounD=runs/round_episode/summary.json \
    inD=runs/ind_episode/summary.json \
  --output_dir runs/compare_episode_all
```
也可以比较同一数据集不同 profile：
```bash
python -m src.responsivegpt.interface.compare_runner \
  --mode episode \
  --items \
    aggressive=runs/highd_episode_aggressive/summary.json \
    balanced=runs/highd_episode_balanced/summary.json \
    conservative=runs/highd_episode_conservative/summary.json \
  --output_dir runs/compare_highd_episode_profiles
```
画图：
```bash
python -m src.responsivegpt.interface.compare_plotter \
  --json_path runs/compare_episode_all/episode_compare.json \
  --output_dir runs/compare_episode_all/plots
```  
  
### 同时支持 batch 和 episode 的对比
同一标签一一对应：：
```bash
python -m src.responsivegpt.interface.compare_runner \
  --mode cross \
  --batch_items \
    highD=runs/highd_batch/summary.json \
    rounD=runs/round_batch/summary.json \
    inD=runs/ind_batch/summary.json \
  --episode_items \
    highD=runs/highd_episode/summary.json \
    rounD=runs/round_episode/summary.json \
    inD=runs/ind_episode/summary.json \
  --output_dir runs/compare_cross_all
```
也可以比较同一数据集不同 profile 的 batch vs episode：
```bash
python -m src.responsivegpt.interface.compare_runner \
  --mode cross \
  --batch_items \
    aggressive=runs/ind_batch_aggressive/summary.json \
    balanced=runs/ind_batch_balanced/summary.json \
    conservative=runs/ind_batch_conservative/summary.json \
  --episode_items \
    aggressive=runs/ind_episode_aggressive/summary.json \
    balanced=runs/ind_episode_balanced/summary.json \
    conservative=runs/ind_episode_conservative/summary.json \
  --output_dir runs/compare_ind_profile_cross
```
画图：
```bash
python -m src.responsivegpt.interface.compare_plotter \
  --json_path runs/compare_cross_all/cross_compare.json \
  --output_dir runs/compare_cross_all/plots
```  

###episode 时序可视化
画某一个 episode
```bash
python -m src.responsivegpt.interface.visualize_episode_timeline \
  --run_dir runs/20260411_highd_episode_conservative \
  --event_index 12
```    
把整个 run 里所有 episode 都画出来
```bash
python -m src.responsivegpt.interface.visualize_episode_timeline \
  --run_dir runs/20260411_highd_episode_conservative
``` 
指定输出目录
```bash
python -m src.responsivegpt.interface.visualize_episode_timeline \
  --run_dir runs/20260411_round_episode_conservative \
  --event_index 5 \
  --output_dir runs/20260411_round_episode_conservative/timeline_figures
``` 

###自动挑选最有代表性的 Top-K episode 并批量画图
```bash
python -m src.responsivegpt.interface.select_and_plot_topk_episodes \
  --run_dir runs/20260411_highd_episode_conservative \
  --top_k 8
``` 
指定输出目录
```bash
python -m src.responsivegpt.interface.select_and_plot_topk_episodes \
  --run_dir runs/20260411_round_episode_conservative \
  --top_k 6 \
  --output_dir runs/20260411_round_episode_conservative/top_cases
```   

###Top-K case report 自动生成
```bash
python -m src.responsivegpt.interface.generate_topk_case_report \
  --run_dir runs/20260411_highd_episode_conservative
``` 
显式指定
```bash
python -m src.responsivegpt.interface.generate_topk_case_report \
  --run_dir runs/20260411_round_episode_conservative \
  --figures_dir runs/20260411_round_episode_conservative/top_cases \
  --selection_json runs/20260411_round_episode_conservative/top_cases/topk_selection.json \
  --output_path runs/20260411_round_episode_conservative/top_cases/topk_case_report.md
```

###自动导出 HTML 版 case report 的模块
```bash
python -m src.responsivegpt.interface.generate_topk_case_report_html \
  --run_dir runs/20260411_highd_episode_conservative
``` 
显式指定
```bash
python -m src.responsivegpt.interface.generate_topk_case_report_html \
  --run_dir runs/20260411_round_episode_conservative \
  --figures_dir runs/20260411_round_episode_conservative/top_cases \
  --selection_json runs/20260411_round_episode_conservative/top_cases/topk_selection.json \
  --output_path runs/20260411_round_episode_conservative/top_cases/topk_case_report.html
```

## 输出

每次运行会自动生成：

`runs/<timestamp>_<tag>/`

包含：

- `config.json`
- `decisions.jsonl`
- `metrics.csv`
- `summary.json`
