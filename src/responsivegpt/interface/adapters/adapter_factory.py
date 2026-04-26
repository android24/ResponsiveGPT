import os

from .highd_event_adapter import HighDEventAdapter
from .round_event_adapter import RoundEventAdapter
from .ind_event_adapter import InDEventAdapter

from .highd_clip_sequence_adapter import HighDClipSequenceAdapter
from .round_clip_sequence_adapter import RoundClipSequenceAdapter
from .ind_scene_sequence_adapter import InDSceneSequenceAdapter


def _clean_path(p: str | None) -> str | None:
    if not p:
        return None
    return str(p).replace("\\", "/").strip()


def _resolve_under_root(root: str, ref: str | None) -> str | None:
    """
    统一处理三种情况：
    1. ref 是绝对路径
    2. ref 是 root 下的相对路径
    3. ref 已经带 clips/ 或 scenes/ 前缀，而 root 已经指向 clips/scenes
    4. ref 只有 basename
    """
    ref = _clean_path(ref)
    if not ref:
        return None

    if os.path.isabs(ref):
        return ref

    root = root or ""

    # 1. root/ref
    p1 = os.path.join(root, ref)
    if os.path.exists(p1):
        return p1

    # 2. 如果 ref 带目录，比如 clips/00/xxx.csv，
    #    而 root 已经是 clips，则去掉第一层再拼
    parts = ref.split("/")
    if len(parts) > 1:
        p2 = os.path.join(root, *parts[1:])
        if os.path.exists(p2):
            return p2

    # 3. basename fallback
    p3 = os.path.join(root, os.path.basename(ref))
    if os.path.exists(p3):
        return p3

    # 返回最可能路径，方便日志打印
    return p1


def build_event_adapter(dataset: str, summary_csv: str):
    dataset = dataset.lower()

    if dataset == "highd":
        return HighDEventAdapter(summary_csv)

    if dataset == "round":
        return RoundEventAdapter(summary_csv)

    if dataset == "ind":
        return InDEventAdapter(summary_csv)

    raise ValueError("dataset must be one of: highd / round / ind")


def build_sequence_adapter(dataset: str, metadata: dict, args):
    """
    返回:
        sequence_adapter, resolved_path, missing_key

    missing_key:
        highD / rounD -> missing_clips
        inD           -> missing_scenes
    """
    dataset = dataset.lower()

    if dataset == "highd":
        clip_ref = metadata.get("clipPath")
        clip_path = _resolve_under_root(args.sequence_root, clip_ref)

        if not clip_path or not os.path.exists(clip_path):
            return None, clip_path, "missing_clips"

        return HighDClipSequenceAdapter(clip_path), clip_path, None

    if dataset == "round":
        clip_ref = metadata.get("clip_file")
        clip_path = _resolve_under_root(args.sequence_root, clip_ref)

        if not clip_path or not os.path.exists(clip_path):
            return None, clip_path, "missing_clips"

        return RoundClipSequenceAdapter(clip_path), clip_path, None

    if dataset == "ind":
        scene_ref = metadata.get("scene_file")
        scene_path = _resolve_under_root(args.sequence_root, scene_ref)

        if not scene_path or not os.path.exists(scene_path):
            return None, scene_path, "missing_scenes"

        return InDSceneSequenceAdapter(scene_path), scene_path, None

    raise ValueError("dataset must be one of: highd / round / ind")