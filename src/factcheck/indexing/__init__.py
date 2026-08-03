"""Offline index build and the online load side, split at the `IndexBuilder` /
`IndexLoader` boundary (see ``ports.py``).

``build.build_index`` is the only thing the CLI needs from build time;
``build.load_retrievers`` is the only thing query time needs. Everything else in this
package is an implementation detail behind those two functions.
"""

from __future__ import annotations

from .build import build_index, load_retrievers
from .manifest import Manifest

__all__ = ["Manifest", "build_index", "load_retrievers"]
