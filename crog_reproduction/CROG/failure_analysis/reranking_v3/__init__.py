"""CROG Full-Chain Evidence Reranker (V3).

V3 is an isolated, frozen-candidate experiment.  It may read immutable V1/V2
artifacts, but every V3 write is constrained to a new ``v3_fullchain_*`` tree.
"""

SCHEMA_VERSION = "3.0.0"
DEFAULT_SEED = 20260801
ENSEMBLE_SEEDS = (20260801, 20260802, 20260803)

