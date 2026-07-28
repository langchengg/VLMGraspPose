# Third-Party Notices

The root `LICENSE` applies to the VLMGraspPose project-owned code. It does not
relicense third-party source trees. Each included or clone-time dependency keeps
its upstream license and copyright notices.

## Included Source Snapshots

| Component | Upstream | Audited commit | License | Local state |
| --- | --- | --- | --- | --- |
| CROG | https://github.com/HilbertXu/CROG | `1eeee85de1fe6bffdc66c9ed9a622028ea04578e` | MIT | Source-only snapshot with Mac/MPS, evaluation, and reranking changes. Data, weights, outputs, and credential-bearing upstream entrypoints are excluded. |
| LAVT-RIS | https://github.com/yz93/LAVT-RIS | `1da0af9f21b637c0cae9ea1363d2dd9b40e19628` | GPL-3.0; embedded `refer/` notices retained | Isolated GPLv3 subtree with OCID-VLG adaptation and local portability changes. |
| VGN | https://github.com/ethz-asl/vgn | `d7af0622433f52ae88ebe81533f12b46b33e951a` | BSD-3-Clause | Unmodified upstream source snapshot from the `corl2020` branch. |

The upstream license files remain inside each included source tree. Local
changes are visible in this repository history and are not claimed as upstream
work.

Audited modification boundaries at import time:

- CROG upstream-tracked changes: `.gitignore`, `engine/crog_engine.py`,
  `utils/dataset.py`, `utils/grasp_eval.py`, and `utils/misc.py`; additional
  Mac/MPS, failure-analysis, evaluation, and reranking files were added locally.
- LAVT upstream-tracked changes: `args.py`, `lib/backbone.py`,
  `lib/segmentation.py`, `test.py`, `train.py`, and `utils.py`; additional
  OCID-VLG configs, loaders, scripts, tests, and portability helpers were added
  locally.
- VGN had no local source modifications when its Git metadata was separated
  from the snapshot.

## Clone-Time Dependencies

These repositories are intentionally not redistributed here. Run
`bash scripts/fetch_external_repositories.sh` to clone an absent dependency at
the audited commit. The script never overwrites an existing checkout.

| Component | Upstream | Audited commit | Reason not included |
| --- | --- | --- | --- |
| HiFi-CS | https://github.com/vineet2104/hifics | `4be6b3be7ce79fae481fb51616adfa2b803f07a0` | No repository-wide license was declared at audit time. |
| GQ-CNN | https://github.com/BerkeleyAutomation/gqcnn | `499a609fe9dfb074bdfb6c4e6e33667ea50f4c21` | Redistribution grant is limited to educational, research, and not-for-profit purposes. |
| GraspNet API / Dex-Net utilities | https://github.com/graspnet/graspnetAPI | `bd6783c3effdebd895abfba8b96dc22a42ec3b5a` | Root MIT license contains embedded subtrees with separate research-only terms. |
| GraspNet baseline | https://github.com/graspnet/graspnet-baseline | `280c215129f759ed8649cb4e89fc5dfee55f4f80` | Upstream terms restrict redistribution and third-party access. |
| VL-Grasp | https://github.com/luyh20/VL-Grasp | `dd6bd6d7b4045b8b72df7d4bebb6ff4a1344076f` | No repository-wide license was declared at audit time. |

Users are responsible for reviewing current upstream terms before downloading,
using, or redistributing any dependency, dataset, or model checkpoint.
