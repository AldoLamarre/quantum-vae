"""Reusable, domain-agnostic loss/regularizer primitives.

Kept separate from trainers/models: functions here take plain tensors
and know nothing about VAEs, quantum circuits, or any specific model.
Consumers (a trainer's compute_loss, a future JEPA-style branch, ...)
own the glue that decides which tensor to call these on and how to
weight the result.
"""
