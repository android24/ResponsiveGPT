# 数据说明
统一知识结构

建议所有 law / case / scenario 都使用一套基础字段：
```bash
{
  "id": "law_ind_vru_yield_001",
  "kb_type": "law",
  "scene_type": "inD",
  "event_type": "car_pedestrian",
  "pair_type": "vehicle_pedestrian",
  "title": "Yield to pedestrian under low spatial margin",
  "condition": {
    "vrus_present": true,
    "ttc_lt": 3.0,
    "dcpa_lt": 2.0,
    "min_future_distance_lt": 2.0
  },
  "risk_mechanism": "Low future distance to a pedestrian creates a short-horizon crossing conflict.",
  "recommended_action": ["yield", "decelerate", "monitor_vru"],
  "forbidden_action": ["accelerate", "aggressive_lane_change"],
  "severity": "high",
  "priority": 0.95,
  "source": "traffic_rulebook",
  "text": "When a vulnerable road user is close to the predicted path, the vehicle should yield and reduce speed."
}
```

## 字段含义
id：唯一证据编号。LLM 输出的 evidence_ids 必须引用它。

kb_type：知识类型，只允许 law / case / scenario。后续可以分别统计法规命中率、案例命中率、场景规则命中率。

scene_type：对应数据集或场景，建议统一为 highD / inD / rounD / all。

event_type：更细的交互类型，比如 car_following、cut_in、car_pedestrian、car_bicycle、roundabout_merging。

pair_type：交互对象类型，比如 vehicle_vehicle、vehicle_pedestrian、vehicle_cyclist。

condition：这条知识生效的物理条件。这个字段很关键，它让 RAG 不只是语义检索，而是能和 TTC、DCPA、DRAC 等指标对齐。

risk_mechanism：解释为什么危险。论文中可以用它支撑可解释性。

recommended_action：允许或推荐的动作，例如 yield、decelerate、increase_headway、monitor_vru。

forbidden_action：禁止或不建议的动作，例如 accelerate、aggressive_lane_change、keep_close_following。

severity / priority：用于排序。法规和 VRU 高风险规则应有更高优先级。

source：来源。可以是法规、人工规则、数据集案例、论文规则、专家标注。

text：给 LLM 看的自然语言证据。

## 三类知识分别怎么写

law 负责合规约束：
```bash
{
  "id": "law_highd_safe_following_001",
  "kb_type": "law",
  "scene_type": "highD",
  "event_type": "car_following",
  "pair_type": "vehicle_vehicle",
  "condition": {
    "ttc_lt": 3.0,
    "thw_lt": 1.2
  },
  "recommended_action": ["increase_headway", "decelerate"],
  "forbidden_action": ["accelerate", "keep_close_following"],
  "severity": "high",
  "priority": 0.9,
  "text": "When following distance or time headway is insufficient, the vehicle should increase headway and avoid acceleration."
}
```

case 负责长尾案例：
```bash
{
  "id": "case_ind_bicycle_close_pass_003",
  "kb_type": "case",
  "scene_type": "inD",
  "event_type": "car_bicycle",
  "pair_type": "vehicle_cyclist",
  "condition": {
    "vrus_present": true,
    "dcpa_lt": 1.5,
    "drac_gt": 6.0
  },
  "risk_mechanism": "A cyclist with low lateral clearance may suddenly deviate, causing near-collision.",
  "recommended_action": ["decelerate", "monitor_vru", "yield"],
  "forbidden_action": ["aggressive_lane_change", "accelerate"],
  "severity": "high",
  "priority": 0.88,
  "source": "inD_high_risk_event",
  "text": "In previous car-bicycle conflicts with low DCPA and high DRAC, early deceleration reduced near-collision risk."
}
```

scenario 负责场景机制：
```bash
{
  "id": "scenario_round_merging_001",
  "kb_type": "scenario",
  "scene_type": "rounD",
  "event_type": "roundabout_merging",
  "pair_type": "vehicle_vehicle",
  "condition": {
    "dcpa_lt": 2.0,
    "min_future_distance_lt": 2.5
  },
  "risk_mechanism": "Roundabout merging conflicts are dominated by future spatial proximity and closest approach distance.",
  "recommended_action": ["yield", "decelerate", "monitor_conflict_point"],
  "forbidden_action": ["accelerate", "force_merge"],
  "severity": "medium",
  "priority": 0.8,
  "text": "For roundabout merging, DCPA and minimum future distance should be prioritized over raw TTC."
}
```