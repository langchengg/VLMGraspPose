from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_crog_engine_has_no_direct_cuda_tensor_transfers():
    source = (REPO_ROOT / "engine" / "crog_engine.py").read_text()

    assert ".cuda(" not in source


def test_grasp_evaluation_avoids_removed_numpy_aliases():
    source = (REPO_ROOT / "utils" / "grasp_eval.py").read_text()

    assert "np.float" not in source
    assert "np.int0" not in source
