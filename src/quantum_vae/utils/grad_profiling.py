"""Gradient-norm profiling for the graph-based quantum interface.

Opt-in HF TrainerCallback: prints grad norm for the projection_kind="graph"
parameters (project_to_quantum's atom-token extractor, qlayer's own pulse
parameters, project_from_quantum's first projection) on the same cadence
as the trainer's normal loss logging (args.logging_steps).

Hooks on_pre_optimizer_step, which fires after backward() and before
optimizer.step()/zero_grad() -- the only point where these gradients are
both populated and untouched.
"""
import warnings
from typing import Optional, Sequence

from transformers import TrainerCallback

DEFAULT_TRACKED_SUBSTRINGS = (
    "project_to_quantum.extractor.atom_queries",
    "qlayer.Omega0_MHz",
    "qlayer.Delta0_MHz",
    "project_from_quantum.hyperedge_proj.weight",
)


class GradNormProfilerCallback(TrainerCallback):
    """Matched by name substring against model.named_parameters() --
    exact names depend on how project_to_quantum/qlayer/project_from_quantum
    nest under the full model.

    Note: assumes args.logging_steps is a positive step count, not the
    <1.0 "fraction of an epoch" form some HF versions also accept -- with
    a fractional value this callback's cadence check will not line up
    with the trainer's own logging.
    """

    def __init__(self, model, param_name_substrings: Optional[Sequence[str]] = None):
        self.model = model
        self.substrings = tuple(param_name_substrings or DEFAULT_TRACKED_SUBSTRINGS)
        self._tracked = None
        self._warned_missing = False

    def _resolve_tracked(self):
        tracked = {
            name: p for name, p in self.model.named_parameters()
            if any(s in name for s in self.substrings)
        }
        if not tracked and not self._warned_missing:
            warnings.warn(
                f"GradNormProfilerCallback matched no parameters for "
                f"{self.substrings} -- check the model uses "
                "projection_kind='graph'.",
                stacklevel=2,
            )
            self._warned_missing = True
        return tracked

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        logging_steps = getattr(args, "logging_steps", 0) or 0
        if logging_steps <= 0 or state.global_step % logging_steps != 0:
            return control

        if self._tracked is None:
            self._tracked = self._resolve_tracked()

        parts = [f"step={state.global_step}"]
        for name, param in self._tracked.items():
            g = param.grad
            norm = g.norm().item() if g is not None else float("nan")
            parts.append(f"{name}={norm:.3e}")

        if len(parts) > 1:
            print("[grad_profile] " + "  ".join(parts))

        return control
