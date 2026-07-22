"""Headless compatibility surface for Berkeley ``visualization==1.0.0``.

The upstream package imports its 3D pyglet viewer eagerly from ``__init__``,
even when GQ-CNN only imports ``Visualizer2D``.  Avoiding that eager import
keeps the CPU scoring container independent of X11 without changing GQ-CNN.
"""

from .visualizer2d import Visualizer2D

__all__ = ["Visualizer2D"]

