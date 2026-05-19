# Architecture Checklist

| Component | Status | Notes |
|---|---|---|
| OCID-VLG loader | COMPLETE | Loads language-conditioned samples with sentence, label, bbox, mask, grasp rectangles. |
| OCID-Grasp loader | COMPLETE | Generates class-based commands when language is missing. |
| Language mapping | COMPLETE | Explicit target-language entries and spatial disambiguation support. |
| Oracle target mode | COMPLETE | Uses dataset bbox/mask without VLM dependencies. |
| VLM target mode | COMPLETE | Local `models/vlm/florence2` loads and one OCID-VLG end-to-end VLM smoke test passed. Grounding quality still needs dataset-level evaluation. |
| RGB-D point cloud | COMPLETE | Open3D conversion, depth scale/truncation, intrinsics fallback. |
| Target point cloud extraction | COMPLETE | Mask extraction with bbox fallback through oracle/VLM target region. |
| Point cloud preprocessing | COMPLETE | Downsampling, outlier removal, table plane, normals, AABB/OBB. |
| Geometric grasp sampler | COMPLETE | Top-down, bbox-aligned, side, normal-based candidates. |
| Candidate validation | COMPLETE | Finite pose, normalized directions/quaternion, width and score range checks. |
| Feature association | COMPLETE | Target overlap, center alignment, width, depth, approach, collision, boundary, grounding score. |
| Rule-based re-ranker | COMPLETE | Configurable semantic-geometric weighted formula. |
| Optional MLP re-ranker | COMPLETE | CPU NumPy MLP scoring head with rule-initialized fallback. |
| Output writer | COMPLETE | Saves mask, PLY, candidates, rankings, best grasp, score breakdown, RGB/3D visualization. |
| Proxy evaluation | COMPLETE | Top-K proxy validity and grouped metrics by dataset, split, scene, target source, and scorer. |
| OCID 2D rectangle evaluation | COMPLETE | Projected rectangle/center metrics. |
| Grounding evaluation | PARTIAL | BBox/mask IoU helpers exist; full batch report integration is minimal. |
| Visualization | COMPLETE | Headless RGB overlay, Top-K projected grasps, 3D matplotlib point cloud plot. |
| CLI scripts | COMPLETE | One-sample, dataset, oracle, VLM, eval, figures, MLP checkpoint helper. |
| Synthetic smoke test | COMPLETE | Included in `tests/test_smoke_pipeline.py`. |
| Real OCID smoke test | COMPLETE | Verified on first OCID-VLG test sample. |
| Mac compatibility | COMPLETE | Core requirements are CPU-compatible; optional VLM requirements are separate. |

Overall status: COMPLETE for oracle/rule-based/optional-MLP CPU experiments and executable VLM mode. Florence-2 grounding quality still needs evaluation/tuning before it should be trusted as the primary target selector.
