# src/responsivegpt/interface/adapters/base_event_adapter.py

from typing import Iterator, Dict, Any
from ...domain.models import SceneState


class BaseEventAdapter:
    """
    所有 summary/event adapter 的统一接口。

    batch 模式:
        row -> SceneState

    episode 模式:
        row -> metadata -> sequence path -> sequence adapter -> SceneState[]
    """

    def iter_rows(self) -> Iterator[dict]:
        raise NotImplementedError

    def row_to_scene(self, row: Dict[str, Any]) -> SceneState:
        """
        用于 batch/event-level baseline。
        一条事件摘要行转成一个 SceneState。
        """
        raise NotImplementedError

    def row_metadata(self, row: Dict[str, Any]) -> dict:
        """
        统一返回事件元信息。
        必须包含后续 resolve sequence path 需要的字段。
        """
        raise NotImplementedError

    def derive_risk_label(self, row: Dict[str, Any]) -> bool:
        """
        数据集弱标签。
        也可以先保留在 evaluation/*.py 中，但放到 adapter 里更内聚。
        """
        raise NotImplementedError

    def get_sequence_ref(self, metadata: dict) -> str | None:
        """
        返回 clip/scene 文件引用。

        highD: clipPath
        rounD: clip_file
        inD: scene_file
        """
        raise NotImplementedError