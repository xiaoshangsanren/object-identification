"""PAD-Lite training and retrieval evaluation for Annotation.

The implementation lives in ``PAD_Lite/src``.  Extending the package search
path keeps the historical entry points (for example
``python -m PAD_Lite.dino_cli``) stable after the source-tree cleanup.
"""

from pathlib import Path


_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if _SOURCE_ROOT.is_dir():
    __path__.append(str(_SOURCE_ROOT))

__version__ = "0.1.0"
