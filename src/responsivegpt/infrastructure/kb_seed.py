from ..domain.evidence import KnowledgeDoc


def default_kb_docs() -> list[KnowledgeDoc]:
    docs = [
        # ======================
        # LAW KB
        # ======================
        KnowledgeDoc(
            id="law_highd_following_distance_001",
            kb_type="law",
            title="高速跟车安全距离规则",
            text="高速行驶场景中，车辆应保持足以避免追尾的安全车距。当相对速度较高、前向间距较小时，应优先减速并避免急促接近前车。",
            scene_type="highD",
            event_type="FOLLOWING_CRITICAL",
            source="traffic_law_kb",
            priority=1.0,
        ),
        KnowledgeDoc(
            id="law_highd_lane_change_gap_001",
            kb_type="law",
            title="高速变道间隙接受规则",
            text="高速变道前应确认目标车道具有充分安全间隙。若后车接近速度较高或相邻车道空间不足，不应实施强行切入。",
            scene_type="highD",
            event_type="CUTIN_CRITICAL",
            source="traffic_law_kb",
            priority=1.0,
        ),
        KnowledgeDoc(
            id="law_round_yield_vehicle_001",
            kb_type="law",
            title="环岛车辆让行规则",
            text="环岛汇入与驶出过程中，驾驶员应关注已在环岛内运行车辆，并根据相对位置和速度主动让行，避免形成危险抢行。",
            scene_type="rounD",
            pair_type="vehicle_vehicle",
            source="traffic_law_kb",
            priority=1.0,
        ),
        KnowledgeDoc(
            id="law_round_yield_cyclist_001",
            kb_type="law",
            title="环岛车辆与骑行者交互规则",
            text="在环岛与非机动车交互中，机动车应保持更高安全裕度，避免以效率为优先进行近距离穿插、抢先通过或压缩骑行者通行空间。",
            scene_type="rounD",
            pair_type="vehicle_cyclist",
            source="traffic_law_kb",
            priority=1.0,
        ),
        KnowledgeDoc(
            id="law_general_vru_priority_001",
            kb_type="law",
            title="弱势道路参与者优先保护原则",
            text="涉及行人或骑行者等弱势道路参与者时，机动车驾驶策略应采用更保守的风险阈值，并优先保障其通行安全。",
            scene_type="all",
            pair_type="vehicle_cyclist",
            source="traffic_law_kb",
            priority=1.0,
        ),

        # ======================
        # CASE KB
        # ======================
        KnowledgeDoc(
            id="case_highd_cutin_rear_collision_001",
            kb_type="case",
            title="高速 cut-in 后方追尾风险案例",
            text="在高速场景中，切入车辆若在较小车头间距下完成变道，会显著增加后方追尾风险。尤其当相对速度为正且最小 TTC 较低时，应判定为高风险交互。",
            scene_type="highD",
            event_type="CUTIN_CRITICAL",
            source="case_kb",
            priority=0.9,
        ),
        KnowledgeDoc(
            id="case_highd_following_low_ttc_001",
            kb_type="case",
            title="高速低 TTC 跟驰风险案例",
            text="在持续跟驰场景中，若前向距离持续偏小且 closing speed 为正，则最小 TTC 会快速下降，此类场景通常对应急刹或追尾近失碰风险。",
            scene_type="highD",
            event_type="FOLLOWING_CRITICAL",
            source="case_kb",
            priority=0.9,
        ),
        KnowledgeDoc(
            id="case_round_vehicle_cyclist_close_001",
            kb_type="case",
            title="环岛机动车与骑行者近距离冲突案例",
            text="当机动车在环岛入口或出口附近与骑行者发生近距离并行、切入或抢先通过时，若最小距离和 TTC 同时较低，应视为高风险并优先采取让行策略。",
            scene_type="rounD",
            pair_type="vehicle_cyclist",
            source="case_kb",
            priority=0.95,
        ),
        KnowledgeDoc(
            id="case_round_vehicle_vehicle_merge_001",
            kb_type="case",
            title="环岛车辆汇入冲突案例",
            text="环岛内外车辆在汇入、汇出和环内并行过程中，如果双方都以维持效率为主而缺少让行，则容易形成低间距、高相对速度的危险交互。",
            scene_type="rounD",
            pair_type="vehicle_vehicle",
            source="case_kb",
            priority=0.9,
        ),

        # ======================
        # SCENARIO KB
        # ======================
        KnowledgeDoc(
            id="scenario_highd_following_pattern_001",
            kb_type="scenario",
            title="高速关键跟驰风险模式",
            text="高速关键跟驰的典型风险模式包括：小车头时距、正向 closing speed、前车减速能力不足、以及驾驶策略未及时提高安全裕度。",
            scene_type="highD",
            event_type="FOLLOWING_CRITICAL",
            source="scenario_rulebook",
            priority=0.95,
        ),
        KnowledgeDoc(
            id="scenario_highd_cutin_pattern_001",
            kb_type="scenario",
            title="高速 cut-in 风险模式",
            text="高速 cut-in 的典型风险模式包括：目标车道间隙不足、切入后 headway 突降、后车反应空间不足、以及驾驶风格偏激进导致的 gap acceptance 过小。",
            scene_type="highD",
            event_type="CUTIN_CRITICAL",
            source="scenario_rulebook",
            priority=0.95,
        ),
        KnowledgeDoc(
            id="scenario_round_merge_pattern_001",
            kb_type="scenario",
            title="环岛汇入与让行风险模式",
            text="环岛场景的主要风险来自进入、环内运行、驶出三类交互重叠。当车辆忽视已在环岛内对象或弱势参与者时，低距离与高冲突概率会快速出现。",
            scene_type="rounD",
            source="scenario_rulebook",
            priority=0.95,
        ),
        KnowledgeDoc(
            id="scenario_round_cyclist_pattern_001",
            kb_type="scenario",
            title="环岛 vehicle-cyclist 风险模式",
            text="机动车与骑行者在环岛交互时，应重点关注相对位置、穿插趋势和通行优先权。当存在抢先通行倾向时，应提高风险等级并触发更保守策略。",
            scene_type="rounD",
            pair_type="vehicle_cyclist",
            source="scenario_rulebook",
            priority=1.0,
        ),
        KnowledgeDoc(
            id="scenario_preference_guardrail_001",
            kb_type="scenario",
            title="驾驶偏好与安全护栏关系",
            text="激进驾驶偏好可以提升效率，但不得突破最低安全约束。保守驾驶偏好应在低 TTC、低间距和弱势参与者出现时显著提高安全权重。",
            scene_type="all",
            source="scenario_rulebook",
            priority=1.0,
        ),
    ]
    return docs