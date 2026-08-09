"""Validation-fitted gates that fail closed to the frozen baseline choice."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reranking.models.tabular import assert_no_forbidden_columns


DEFAULT_GATE_FEATURES = (
    "challenger_margin",
    "baseline_score_delta",
    "baseline_rank_delta",
    "mask_support_delta",
    "width_compatibility_delta",
    "depth_contact_delta",
    "collision_proxy_delta",
    "challenger_risk",
    "challenger_reliability",
    "model_disagreement",
    "seed_agreement",
    "score_perturbation_stability",
)
PAIR_SIGNAL_COLUMNS = {
    "baseline_rank": "baseline_rank_delta",
    "mask_support_signal": "mask_support_delta",
    "width_compatibility_signal": "width_compatibility_delta",
    "depth_contact_signal": "depth_contact_delta",
    "collision_proxy_signal": "collision_proxy_delta",
}
ALLOWED_FIT_SCOPES = frozenset({"validation", "oof"})


class GateError(ValueError):
    """Raised when gate data or fit provenance violates the contract."""


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise GateError(f"gate table missing columns: {missing}")


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise GateError(f"{column} must be finite numeric values")
    return values


def _validate_fit_scope(examples: pd.DataFrame, fit_scope: str) -> str:
    scope = str(fit_scope).strip().lower()
    if scope not in ALLOWED_FIT_SCOPES:
        raise GateError("gate fit_scope must be explicitly 'validation' or 'oof'")
    if "split" in examples.columns:
        observed = set(examples["split"].astype(str).str.lower().unique())
        if observed & {"test", "testing", "final_test", "holdout"}:
            raise GateError(f"test/holdout rows cannot fit a gate: {sorted(observed)}")
        if scope == "validation" and observed != {"validation"}:
            raise GateError(
                "validation gate fit requires every split value to be validation"
            )
    if scope == "oof":
        if "oof_fold" not in examples.columns:
            raise GateError("OOF gate fit requires an explicit oof_fold column")
        folds = pd.to_numeric(examples["oof_fold"], errors="coerce")
        if folds.isna().any() or bool((folds < 0).any()):
            raise GateError("oof_fold must contain non-negative fold identifiers")
    return scope


def _outcome_values(outcomes: Sequence[Any], *, length: int) -> np.ndarray:
    values = np.asarray(outcomes, dtype=np.float64)
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise GateError("outcomes must be one finite value per gate example")
    if not set(np.unique(values)).issubset({-1.0, 0.0, 1.0}):
        raise GateError("outcomes must be -1 (harmful), 0 (neutral), or 1 (beneficial)")
    return values


def _feature_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    _require_columns(frame, columns)
    assert_no_forbidden_columns(columns)
    try:
        matrix = frame.loc[:, columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GateError("gate features must be numeric") from exc
    if not np.all(np.isfinite(matrix)):
        raise GateError("gate features must be finite")
    return matrix


def _proposal_changes(examples: pd.DataFrame) -> np.ndarray:
    _require_columns(examples, ("baseline_candidate_id", "challenger_candidate_id"))
    old = examples["baseline_candidate_id"]
    new = examples["challenger_candidate_id"]
    valid = old.notna() & new.notna()
    old_string = old.fillna("").astype(str)
    new_string = new.fillna("").astype(str)
    valid &= old_string.ne("") & new_string.ne("")
    return (valid & old_string.ne(new_string)).to_numpy(dtype=bool)


def _fit_ids(examples: pd.DataFrame) -> list[str]:
    if "query_id" in examples.columns:
        return examples["query_id"].astype(str).tolist()
    return [str(index) for index in range(len(examples))]


def _choose_threshold(
    signal: np.ndarray,
    outcomes: np.ndarray,
    *,
    eligible: np.ndarray,
    harmful_rate_limit: float,
    minimum_threshold: float = -np.inf,
) -> tuple[float, list[dict[str, Any]]]:
    if not 0.0 <= harmful_rate_limit <= 1.0:
        raise GateError("harmful_rate_limit must be between zero and one")
    if signal.shape != outcomes.shape or eligible.shape != outcomes.shape:
        raise GateError("threshold inputs must have matching shapes")
    if not np.all(np.isfinite(signal)):
        raise GateError("threshold signal must be finite")
    candidates = sorted(
        {
            float(value)
            for value in signal
            if float(value) >= float(minimum_threshold)
        },
        reverse=True,
    )
    if np.isfinite(minimum_threshold):
        candidates.append(float(minimum_threshold))
    maximum = max(candidates, default=float(minimum_threshold))
    no_switch = float(np.nextafter(maximum, np.inf))
    candidates = sorted(set([*candidates, no_switch]), reverse=True)

    eligible_indices = np.flatnonzero(eligible)
    eligible_order = eligible_indices[
        np.argsort(-signal[eligible_indices], kind="mergesort")
    ]
    cursor = 0
    beneficial = 0
    harmful = 0
    neutral = 0
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        while (
            cursor < len(eligible_order)
            and signal[eligible_order[cursor]] >= threshold
        ):
            outcome = outcomes[eligible_order[cursor]]
            beneficial += int(outcome > 0.0)
            harmful += int(outcome < 0.0)
            neutral += int(outcome == 0.0)
            cursor += 1
        harmful_rate = harmful / max(len(outcomes), 1)
        rows.append(
            {
                "threshold": float(threshold),
                "switch_count": int(beneficial + harmful + neutral),
                "beneficial": beneficial,
                "harmful": harmful,
                "neutral": neutral,
                "net_gain": beneficial - harmful,
                "harmful_rate": float(harmful_rate),
                "eligible_under_harm_limit": harmful_rate <= harmful_rate_limit,
            }
        )
    feasible = [row for row in rows if row["eligible_under_harm_limit"]]
    selected = max(
        feasible,
        key=lambda row: (
            row["net_gain"],
            -row["harmful"],
            -row["switch_count"],
            row["threshold"],
        ),
    )
    # Preserve an exact selected threshold while bounding checkpoint metadata.
    if len(rows) > 512:
        stride = int(np.ceil(len(rows) / 500))
        retained = rows[::stride]
        retained.extend(
            row
            for row in rows
            if row["threshold"] in {selected["threshold"], no_switch}
        )
        rows = sorted(
            {row["threshold"]: row for row in retained}.values(),
            key=lambda row: row["threshold"],
            reverse=True,
        )
    return float(selected["threshold"]), rows


def build_switch_proposals(
    candidates: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
    baseline_score_col: str = "baseline_score",
    reranker_score_col: str = "reranker_score",
    risk_col: str = "risk",
    reliability_col: str = "reliability",
) -> pd.DataFrame:
    """Freeze one baseline/challenger proposal per query without labels.

    Exact score ties are resolved by candidate ID ascending for both methods.
    """

    required = (
        query_col,
        candidate_col,
        baseline_score_col,
        reranker_score_col,
        risk_col,
        reliability_col,
    )
    _require_columns(candidates, required)
    optional = [column for column in PAIR_SIGNAL_COLUMNS if column in candidates.columns]
    frame = candidates.loc[:, (*required, *optional)].copy()
    for column in (query_col, candidate_col):
        if frame[column].isna().any():
            raise GateError(f"{column} contains null values")
        frame[column] = frame[column].astype(str)
    if frame.duplicated([query_col, candidate_col]).any():
        raise GateError("duplicate query/candidate identity")
    for column in (baseline_score_col, reranker_score_col, risk_col, reliability_col):
        frame[column] = _finite_column(frame, column)
    for column in optional:
        frame[column] = _finite_column(frame, column)

    rows: list[dict[str, Any]] = []
    for query_id, group in frame.groupby(query_col, sort=True):
        baseline_order = group.sort_values(
            [baseline_score_col, candidate_col],
            ascending=[False, True],
            kind="mergesort",
        )
        challenger_order = group.sort_values(
            [reranker_score_col, candidate_col],
            ascending=[False, True],
            kind="mergesort",
        )
        baseline = baseline_order.iloc[0]
        challenger = challenger_order.iloc[0]
        reranker_values = challenger_order[reranker_score_col].to_numpy(np.float64)
        margin = (
            float(reranker_values[0] - reranker_values[1])
            if len(reranker_values) > 1
            else 0.0
        )
        proposal_changed = str(baseline[candidate_col]) != str(challenger[candidate_col])
        row = {
                "query_id": str(query_id),
                "baseline_candidate_id": str(baseline[candidate_col]),
                "challenger_candidate_id": str(challenger[candidate_col]),
                "proposal_changes": bool(proposal_changed),
                "challenger_margin": margin,
                "baseline_score": float(baseline[baseline_score_col]),
                "challenger_baseline_score": float(challenger[baseline_score_col]),
                "baseline_score_delta": float(
                    challenger[baseline_score_col] - baseline[baseline_score_col]
                ),
                "challenger_score": float(challenger[reranker_score_col]),
                "challenger_risk": float(challenger[risk_col]),
                "challenger_reliability": float(challenger[reliability_col]),
                "model_disagreement": float(proposal_changed),
                "seed_agreement": 1.0,
                "score_perturbation_stability": float(
                    margin
                    > 0.02
                    * max(abs(float(challenger[reranker_score_col])), 1e-3)
                ),
            }
        for source, destination in PAIR_SIGNAL_COLUMNS.items():
            row[destination] = (
                float(challenger[source] - baseline[source])
                if source in optional
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def attach_switch_outcomes(
    proposals: pd.DataFrame,
    labelled_candidates: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
    label_col: str = "label",
) -> pd.DataFrame:
    """Join validation labels after label-free proposals have been frozen."""

    _require_columns(
        proposals, ("query_id", "baseline_candidate_id", "challenger_candidate_id")
    )
    _require_columns(labelled_candidates, (query_col, candidate_col, label_col))
    labels = labelled_candidates.loc[:, (query_col, candidate_col, label_col)].copy()
    if labels.duplicated([query_col, candidate_col]).any():
        raise GateError("duplicate query/candidate labels")
    values = pd.to_numeric(labels[label_col], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise GateError(f"{label_col} must be binary")
    index = {
        (str(query), str(candidate)): bool(label)
        for query, candidate, label in labels.itertuples(index=False, name=None)
    }
    output = proposals.copy()
    outcomes: list[int] = []
    for row in output.itertuples(index=False):
        old_key = (str(row.query_id), str(row.baseline_candidate_id))
        new_key = (str(row.query_id), str(row.challenger_candidate_id))
        if old_key not in index or new_key not in index:
            raise GateError(f"proposal candidate label missing for query {row.query_id}")
        old_correct = index[old_key]
        new_correct = index[new_key]
        outcomes.append(
            1 if (not old_correct and new_correct) else -1 if (old_correct and not new_correct) else 0
        )
    output["switch_outcome"] = np.asarray(outcomes, dtype=np.int8)
    return output


def _apply_decisions(
    examples: pd.DataFrame,
    switch: np.ndarray,
    reason: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
) -> pd.DataFrame:
    result = examples.copy()
    result["switch_applied"] = switch.astype(bool)
    result["selected_candidate_id"] = np.where(
        switch,
        result["challenger_candidate_id"].astype(str),
        result["baseline_candidate_id"].astype(str),
    )
    result["fallback_reason"] = reason
    if confidence is not None:
        result["estimated_benefit_probability"] = confidence
    return result


class MarginGate:
    """Validation-selected margin, reliability, and optional q-floor gate."""

    def __init__(
        self,
        *,
        margin_col: str = "challenger_margin",
        reliability_col: str = "challenger_reliability",
        baseline_score_col: str = "baseline_score",
        search_reliability: bool = True,
        search_baseline_floor: bool = True,
        harmful_rate_limit: float = 0.01,
    ) -> None:
        self.margin_col = str(margin_col)
        self.reliability_col = str(reliability_col)
        self.baseline_score_col = str(baseline_score_col)
        self.search_reliability = bool(search_reliability)
        self.search_baseline_floor = bool(search_baseline_floor)
        self.harmful_rate_limit = float(harmful_rate_limit)

    def fit(
        self,
        examples: pd.DataFrame,
        outcomes: Sequence[Any],
        *,
        fit_scope: str,
    ) -> "MarginGate":
        self.fit_scope_ = _validate_fit_scope(examples, fit_scope)
        _require_columns(examples, (self.margin_col, "baseline_candidate_id", "challenger_candidate_id"))
        margin = _finite_column(examples, self.margin_col)
        outcome = _outcome_values(outcomes, length=len(examples))
        changed = _proposal_changes(examples)
        reliability = (
            _finite_column(examples, self.reliability_col)
            if self.reliability_col in examples.columns
            else np.ones(len(examples), dtype=np.float64)
        )
        baseline = (
            _finite_column(examples, self.baseline_score_col)
            if self.baseline_score_col in examples.columns
            else np.zeros(len(examples), dtype=np.float64)
        )
        reliability_thresholds = [float("-inf")]
        if self.search_reliability and self.reliability_col in examples.columns:
            reliability_thresholds.extend(
                map(float, np.quantile(reliability, [0.25, 0.5, 0.75]))
            )
        baseline_floors = [float("-inf")]
        if self.search_baseline_floor and self.baseline_score_col in examples.columns:
            baseline_floors.extend(
                map(float, np.quantile(baseline, [0.05, 0.25]))
            )
        searches: list[dict[str, Any]] = []
        for reliability_threshold in sorted(set(reliability_thresholds)):
            for baseline_floor in sorted(set(baseline_floors)):
                threshold, sweep = _choose_threshold(
                    margin,
                    outcome,
                    eligible=(
                        changed
                        & (reliability >= reliability_threshold)
                        & (baseline >= baseline_floor)
                    ),
                    harmful_rate_limit=self.harmful_rate_limit,
                )
                selected = next(
                    row for row in sweep if row["threshold"] == threshold
                )
                searches.append(
                    {
                        **selected,
                        "reliability_threshold": reliability_threshold,
                        "baseline_score_floor": baseline_floor,
                        "margin_sweep": sweep,
                    }
                )
        selected_search = max(
            searches,
            key=lambda row: (
                row["net_gain"],
                -row["harmful"],
                -row["switch_count"],
                row["reliability_threshold"],
                row["baseline_score_floor"],
                row["threshold"],
            ),
        )
        self.threshold_ = float(selected_search["threshold"])
        self.reliability_threshold_ = float(
            selected_search["reliability_threshold"]
        )
        self.baseline_score_floor_ = float(
            selected_search["baseline_score_floor"]
        )
        self.threshold_audit_ = searches
        self.fit_query_ids_ = _fit_ids(examples)
        return self

    def predict_switch(self, examples: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("margin gate has not been fitted")
        margin = _finite_column(examples, self.margin_col)
        reliability = (
            _finite_column(examples, self.reliability_col)
            if self.reliability_col in examples.columns
            else np.ones(len(examples), dtype=np.float64)
        )
        baseline = (
            _finite_column(examples, self.baseline_score_col)
            if self.baseline_score_col in examples.columns
            else np.zeros(len(examples), dtype=np.float64)
        )
        return (
            _proposal_changes(examples)
            & (margin >= self.threshold_)
            & (reliability >= self.reliability_threshold_)
            & (baseline >= self.baseline_score_floor_)
        )

    def apply(self, examples: pd.DataFrame) -> pd.DataFrame:
        switch = self.predict_switch(examples)
        changed = _proposal_changes(examples)
        reason = np.where(
            switch,
            "switch",
            np.where(~changed, "same_candidate", "below_margin_threshold"),
        )
        return _apply_decisions(examples, switch, reason)

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("margin gate has not been fitted")
        return {
            "gate_kind": "validation_selected_margin",
            "fit_scope": self.fit_scope_,
            "fit_query_ids": list(self.fit_query_ids_),
            "margin_column": self.margin_col,
            "threshold": self.threshold_,
            "reliability_column": self.reliability_col,
            "reliability_threshold": self.reliability_threshold_,
            "baseline_score_column": self.baseline_score_col,
            "baseline_score_floor": self.baseline_score_floor_,
            "harmful_rate_limit": self.harmful_rate_limit,
            "threshold_tie_break": "highest threshold after net/harm/coverage",
        }


class LearnedLogisticGate:
    """Logistic estimate of a beneficial switch, fitted on validation/OOF only."""

    def __init__(
        self,
        feature_columns: Sequence[str] = DEFAULT_GATE_FEATURES,
        *,
        c: float = 1.0,
        harmful_rate_limit: float = 0.01,
        random_state: int = 0,
    ) -> None:
        self.feature_columns = tuple(map(str, feature_columns))
        assert_no_forbidden_columns(self.feature_columns)
        self.c = float(c)
        self.harmful_rate_limit = float(harmful_rate_limit)
        self.random_state = int(random_state)

    def fit(
        self,
        examples: pd.DataFrame,
        outcomes: Sequence[Any],
        *,
        fit_scope: str,
    ) -> "LearnedLogisticGate":
        self.fit_scope_ = _validate_fit_scope(examples, fit_scope)
        matrix = _feature_matrix(examples, self.feature_columns)
        outcome = _outcome_values(outcomes, length=len(examples))
        decisive = _proposal_changes(examples) & (outcome != 0.0)
        fit_matrix = matrix[decisive]
        target = (outcome[decisive] > 0.0).astype(np.int8)
        self.scaler_ = StandardScaler().fit(
            fit_matrix if len(fit_matrix) else matrix
        )
        self.decisive_fit_count_ = int(decisive.sum())
        self.neutral_ignored_count_ = int((outcome == 0.0).sum())
        if len(target) == 0:
            self.model_ = None
            self.constant_probability_ = 0.0
        elif len(np.unique(target)) == 1:
            self.model_ = None
            self.constant_probability_ = float(target[0])
        else:
            self.model_ = LogisticRegression(
                C=self.c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=self.random_state,
            )
            self.model_.fit(self.scaler_.transform(fit_matrix), target)
            self.constant_probability_ = None
        probability = self.predict_gain_probability(examples)
        self.threshold_, self.threshold_audit_ = _choose_threshold(
            probability,
            outcome,
            eligible=_proposal_changes(examples),
            harmful_rate_limit=self.harmful_rate_limit,
        )
        self.fit_query_ids_ = _fit_ids(examples)
        return self

    def predict_gain_probability(self, examples: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("learned gate has not been fitted")
        matrix = _feature_matrix(examples, self.feature_columns)
        if self.model_ is None:
            probability = np.full(len(examples), self.constant_probability_, dtype=np.float64)
        else:
            probability = self.model_.predict_proba(self.scaler_.transform(matrix))[:, 1]
        if not np.all(np.isfinite(probability)):
            raise RuntimeError("learned gate produced non-finite probabilities")
        return np.asarray(probability, dtype=np.float64)

    def predict_switch(self, examples: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("learned gate has not been fitted")
        probability = self.predict_gain_probability(examples)
        return _proposal_changes(examples) & (probability >= self.threshold_)

    def apply(self, examples: pd.DataFrame) -> pd.DataFrame:
        probability = self.predict_gain_probability(examples)
        changed = _proposal_changes(examples)
        switch = changed & (probability >= self.threshold_)
        reason = np.where(
            switch,
            "switch",
            np.where(~changed, "same_candidate", "benefit_below_threshold"),
        )
        return _apply_decisions(examples, switch, reason, confidence=probability)

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("learned gate has not been fitted")
        return {
            "gate_kind": "validation_only_logistic_benefit",
            "fit_scope": self.fit_scope_,
            "fit_query_ids": list(self.fit_query_ids_),
            "feature_columns": list(self.feature_columns),
            "threshold": self.threshold_,
            "harmful_rate_limit": self.harmful_rate_limit,
            "model": "constant" if self.model_ is None else "LogisticRegression",
            "neutral_outcomes": "ignored_for_logistic_fit_retained_for_threshold_audit",
            "decisive_fit_count": self.decisive_fit_count_,
            "neutral_ignored_count": self.neutral_ignored_count_,
            "threshold_tie_break": "highest threshold after net/harm/coverage",
        }


class ConservativeGate:
    """Learned benefit gate intersected with preregistered risk constraints."""

    def __init__(
        self,
        feature_columns: Sequence[str] = DEFAULT_GATE_FEATURES,
        *,
        min_margin: float = 0.0,
        max_risk: float = 0.0,
        min_reliability: float = 0.0,
        min_baseline_score: float | None = None,
        minimum_gain_probability: float = 0.5,
        harmful_rate_limit: float = 0.01,
        search_constraints: bool = False,
        c: float = 1.0,
        random_state: int = 0,
    ) -> None:
        if not 0.0 <= minimum_gain_probability <= 1.0:
            raise GateError("minimum_gain_probability must be between zero and one")
        self.feature_columns = tuple(map(str, feature_columns))
        self.min_margin = float(min_margin)
        self.max_risk = float(max_risk)
        self.min_reliability = float(min_reliability)
        self.min_baseline_score = (
            None if min_baseline_score is None else float(min_baseline_score)
        )
        self.minimum_gain_probability = float(minimum_gain_probability)
        self.harmful_rate_limit = float(harmful_rate_limit)
        self.search_constraints = bool(search_constraints)
        self.c = float(c)
        self.random_state = int(random_state)

    def _constraints(self, examples: pd.DataFrame) -> np.ndarray:
        required = ["challenger_margin", "challenger_risk", "challenger_reliability"]
        if self.min_baseline_score is not None:
            required.append("baseline_score")
        _require_columns(examples, required)
        allowed = (
            _proposal_changes(examples)
            & (_finite_column(examples, "challenger_margin") >= self.min_margin)
            & (_finite_column(examples, "challenger_risk") <= self.max_risk)
            & (_finite_column(examples, "challenger_reliability") >= self.min_reliability)
        )
        if self.min_baseline_score is not None:
            allowed &= _finite_column(examples, "baseline_score") >= self.min_baseline_score
        return allowed

    def fit(
        self,
        examples: pd.DataFrame,
        outcomes: Sequence[Any],
        *,
        fit_scope: str,
    ) -> "ConservativeGate":
        self.learned_gate_ = LearnedLogisticGate(
            self.feature_columns,
            c=self.c,
            harmful_rate_limit=self.harmful_rate_limit,
            random_state=self.random_state,
        ).fit(examples, outcomes, fit_scope=fit_scope)
        probability = self.learned_gate_.predict_gain_probability(examples)
        outcome = _outcome_values(outcomes, length=len(examples))
        self.constraint_search_audit_: list[dict[str, Any]] = []
        if self.search_constraints:
            changed = _proposal_changes(examples)
            margin = _finite_column(examples, "challenger_margin")
            risk = _finite_column(examples, "challenger_risk")
            reliability = _finite_column(examples, "challenger_reliability")
            baseline = _finite_column(examples, "baseline_score")

            def select_constraint(
                attribute: str,
                candidates: Sequence[float | None],
            ) -> None:
                rows: list[dict[str, Any]] = []
                initial = getattr(self, attribute)
                for value in candidates:
                    setattr(self, attribute, value)
                    switch = self._constraints(examples) & (
                        probability >= self.minimum_gain_probability
                    )
                    beneficial = int(np.sum(switch & (outcome > 0.0)))
                    harmful = int(np.sum(switch & (outcome < 0.0)))
                    rows.append(
                        {
                            "value": value,
                            "switch_count": int(switch.sum()),
                            "beneficial": beneficial,
                            "harmful": harmful,
                            "net_gain": beneficial - harmful,
                        }
                    )
                selected = max(
                    rows,
                    key=lambda row: (
                        row["net_gain"],
                        -row["harmful"],
                        -row["switch_count"],
                    ),
                )
                setattr(self, attribute, selected["value"])
                self.constraint_search_audit_.append(
                    {
                        "constraint": attribute,
                        "initial": initial,
                        "selected": selected["value"],
                        "grid": rows,
                    }
                )

            if changed.any():
                select_constraint(
                    "min_margin",
                    sorted(
                        set(
                            [
                                0.0,
                                *map(
                                    float,
                                    np.quantile(
                                        margin[changed], [0.25, 0.5, 0.75]
                                    ),
                                ),
                            ]
                        )
                    ),
                )
                select_constraint(
                    "max_risk",
                    sorted(
                        set(
                            map(
                                float,
                                np.quantile(
                                    risk[changed], [0.25, 0.5, 0.75, 1.0]
                                ),
                            )
                        )
                    ),
                )
                select_constraint(
                    "min_reliability",
                    sorted(
                        set(
                            [
                                0.0,
                                *map(
                                    float,
                                    np.quantile(
                                        reliability[changed], [0.25, 0.5, 0.75]
                                    ),
                                ),
                            ]
                        )
                    ),
                )
                select_constraint(
                    "min_baseline_score",
                    [
                        None,
                        *map(
                            float,
                            np.quantile(baseline[changed], [0.05, 0.25]),
                        ),
                    ],
                )
        self.threshold_, self.threshold_audit_ = _choose_threshold(
            probability,
            outcome,
            eligible=self._constraints(examples),
            harmful_rate_limit=self.harmful_rate_limit,
            minimum_threshold=self.minimum_gain_probability,
        )
        self.fit_scope_ = self.learned_gate_.fit_scope_
        self.fit_query_ids_ = _fit_ids(examples)
        return self

    def predict_gain_probability(self, examples: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "learned_gate_"):
            raise RuntimeError("conservative gate has not been fitted")
        return self.learned_gate_.predict_gain_probability(examples)

    def predict_switch(self, examples: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("conservative gate has not been fitted")
        probability = self.predict_gain_probability(examples)
        return self._constraints(examples) & (probability >= self.threshold_)

    def apply(self, examples: pd.DataFrame) -> pd.DataFrame:
        probability = self.predict_gain_probability(examples)
        changed = _proposal_changes(examples)
        margin_ok = _finite_column(examples, "challenger_margin") >= self.min_margin
        risk_ok = _finite_column(examples, "challenger_risk") <= self.max_risk
        reliability_ok = (
            _finite_column(examples, "challenger_reliability") >= self.min_reliability
        )
        baseline_ok = np.ones(len(examples), dtype=bool)
        if self.min_baseline_score is not None:
            baseline_ok = (
                _finite_column(examples, "baseline_score") >= self.min_baseline_score
            )
        benefit_ok = probability >= self.threshold_
        switch = changed & benefit_ok & margin_ok & risk_ok & reliability_ok & baseline_ok
        reason = np.select(
            [
                ~changed,
                ~benefit_ok,
                ~margin_ok,
                ~risk_ok,
                ~reliability_ok,
                ~baseline_ok,
            ],
            [
                "same_candidate",
                "benefit_below_threshold",
                "margin_below_threshold",
                "risk_above_limit",
                "reliability_below_threshold",
                "baseline_below_floor",
            ],
            default="switch",
        )
        return _apply_decisions(examples, switch, reason, confidence=probability)

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("conservative gate has not been fitted")
        return {
            "gate_kind": "conservative_validation_only_logistic",
            "fit_scope": self.fit_scope_,
            "fit_query_ids": list(self.fit_query_ids_),
            "benefit_probability_threshold": self.threshold_,
            "minimum_gain_probability": self.minimum_gain_probability,
            "min_margin": self.min_margin,
            "max_risk": self.max_risk,
            "min_reliability": self.min_reliability,
            "min_baseline_score": self.min_baseline_score,
            "harmful_rate_limit": self.harmful_rate_limit,
            "selection_rule": "benefit AND changed AND margin AND risk AND reliability AND baseline floor",
            "constraints_selected_on_validation": self.search_constraints,
            "constraint_search_audit": self.constraint_search_audit_,
            "learned_gate": self.learned_gate_.metadata,
        }


__all__ = [
    "ALLOWED_FIT_SCOPES",
    "ConservativeGate",
    "DEFAULT_GATE_FEATURES",
    "GateError",
    "LearnedLogisticGate",
    "MarginGate",
    "attach_switch_outcomes",
    "build_switch_proposals",
]
