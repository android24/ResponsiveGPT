from copy import deepcopy


class TriggerStateStore:
    """
    维护 active triggers 的生命周期。
    """

    def __init__(self):
        self.active_triggers = []

    def add(self, triggers: list):
        self.active_triggers.extend(triggers)

    def tick_frame(self):
        kept = []
        for t in self.active_triggers:
            ttl_frames = max(0, t.ttl_frames - 1)
            ttl_episodes = t.ttl_episodes
            if ttl_frames > 0 or ttl_episodes > 0:
                kept.append(self._replace_trigger(t, ttl_frames=ttl_frames, ttl_episodes=ttl_episodes))
        self.active_triggers = kept

    def tick_episode(self):
        kept = []
        for t in self.active_triggers:
            ttl_frames = t.ttl_frames
            ttl_episodes = max(0, t.ttl_episodes - 1)
            if ttl_frames > 0 or ttl_episodes > 0:
                kept.append(self._replace_trigger(t, ttl_frames=ttl_frames, ttl_episodes=ttl_episodes))
        self.active_triggers = kept

    def get_active(self):
        return list(self.active_triggers)

    def _replace_trigger(self, t, ttl_frames, ttl_episodes):
        return type(t)(
            trigger_id=t.trigger_id,
            trigger_type=t.trigger_type,
            level=t.level,
            source=t.source,
            activated=t.activated,
            score=t.score,
            reason=t.reason,
            action=t.action,
            action_value=t.action_value,
            target_layer=getattr(t, "target_layer", None),
            parameter_key=getattr(t, "parameter_key", None),
            priority=getattr(t, "priority", 0),
            ttl_frames=ttl_frames,
            ttl_episodes=ttl_episodes,
            scene_type=t.scene_type,
            event_type=t.event_type,
            frame_index=t.frame_index,
            metadata=deepcopy(t.metadata),
        )