from __future__ import annotations

from evaluation.evaluator import OutputEvaluator


class ProxyEvaluator(OutputEvaluator):
    def evaluate_records(self, records: list[dict], mode: str = "proxy") -> dict:
        return super().evaluate_records(records, mode="proxy")
