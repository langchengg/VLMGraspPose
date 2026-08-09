from unified_reranking.statistics import (
    cluster_bootstrap_difference,
    mcnemar_exact,
    three_system_paired_tests,
)


def test_three_system_tests_report_all_holm_pairs():
    result = three_system_paired_tests(
        {
            "CROG": [1, 1, 0, 0, 1, 0],
            "G1": [1, 0, 1, 0, 1, 0],
            "C1": [0, 1, 1, 0, 1, 0],
        }
    )
    assert len(result["pairwise_mcnemar_holm"]) == 3
    assert all(0 <= row["holm_adjusted_pvalue"] <= 1 for row in result["pairwise_mcnemar_holm"])


def test_local_cluster_bootstrap_is_paired_and_deterministic():
    reference = [0, 1, 0, 1, 0, 1]
    challenger = [1, 1, 0, 0, 1, 1]
    clusters = ["a", "a", "b", "b", "c", "c"]
    first = cluster_bootstrap_difference(
        reference, challenger, clusters, iterations=200, seed=20260808
    )
    second = cluster_bootstrap_difference(
        reference, challenger, clusters, iterations=200, seed=20260808
    )
    assert first == second
    assert first["point_estimate"] == (2 - 1) / 6
    assert first["cluster_count"] == 3
    exact = mcnemar_exact(reference, challenger)
    assert exact["recovered"] == 2
    assert exact["harmful"] == 1


def test_identical_systems_have_finite_degenerate_cochran_q() -> None:
    result = three_system_paired_tests(
        {"CROG": [1, 0, 1], "G1": [1, 0, 1], "C1": [1, 0, 1]}
    )["cochran_q"]
    assert result["statistic"] == 0.0
    assert result["pvalue"] == 1.0
    assert result["degenerate_identical_systems"] is True
