"""CPU refinement visualization façade.

The plotting implementation is device-independent and is reused verbatim.
"""

from .sam3_visualization import (
    save_candidate_grid,
    save_coarse_vs_refined,
    save_mask_overlay,
    save_prompt_visualization,
)

__all__ = [
    "save_candidate_grid",
    "save_coarse_vs_refined",
    "save_mask_overlay",
    "save_prompt_visualization",
]
