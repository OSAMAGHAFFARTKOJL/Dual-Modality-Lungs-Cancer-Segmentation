"""Torch-free by design: `metrics`, `spec` and `viz` are safe to import anywhere.

`inference`, `explain` and `model` pull in torch and timm, so they are imported only
on the path that actually runs the network (a user-supplied slice).
"""
from . import metrics, spec, viz

__all__ = ["metrics", "spec", "viz"]
