from dataclasses import dataclass, field
import cv2
from pathlib import Path
from collections.abc import Callable

@dataclass
class SaveAttackContext:
    attacks: dict[str, cv2.typing.MatLike]
    output_dir: Path
    extract_function: Callable
    extract_parameters: dict = field(default_factory=dict)