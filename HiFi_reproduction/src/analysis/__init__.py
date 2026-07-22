"""Analysis-only tools for VGN candidate multiplicity and target consistency.

Nothing in this package modifies VGN scores, candidate pools, or baseline
outputs.  The package consumes frozen experiment artifacts and writes derived
diagnostics under a separate analysis directory.
"""

ANALYSIS_SCHEMA_VERSION = 1

__all__ = ["ANALYSIS_SCHEMA_VERSION"]
