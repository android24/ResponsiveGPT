# src/responsivegpt/interface/adapters/base_sequence_adapter.py

from typing import Iterator
from ...domain.models import SceneState


class BaseSequenceAdapter:
    """
    所有 sequence adapter 的统一接口。
    """

    def iter_scenes(self) -> Iterator[SceneState]:
        raise NotImplementedError("Subclasses must implement iter_scenes().")

    def sequence_metadata(self) -> dict:
        """
        可选：返回 clip/scene 层面的元信息。
        """
        return {}