"""Feature-selection domain errors (Phase F)."""


class LayerMismatchError(ValueError):
    """Raised when decoder/latents/scales/gradients come from different layers."""


class ScopeMismatchError(ValueError):
    """Raised when attribution scope differs from intervention scope."""


class HookSiteMismatchError(ValueError):
    """Raised when the gradient tensor is not the intervention hook site."""
