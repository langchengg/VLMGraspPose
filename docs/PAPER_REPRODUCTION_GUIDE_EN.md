# VLMGraspPose Complete Paper Reproduction Guide (English)

> Paper/dissertation: **From Prediction to Selection: Re-Ranking Grasp Candidates in End-to-End and Modular Pipelines for Language-Guided Robotic Grasping**  
> Repository: [langchengg/VLMGraspPose](https://github.com/langchengg/VLMGraspPose)  
> Audit baseline for this guide: `461505999123fe23303bab45bfc7ddbf9619f1fb`  
> Audit date: 2026-08-27  
> Current reproduction status: **PARTIALLY VERIFIED**

## 0. Read This First: What “Complete Reproduction” Means Here

The formal contribution of the final dissertation in this repository is neither direct RGB-D-to-robot-executable 6-DoF pose prediction nor a reproduction of the physical-robot success rate reported in the original CROG paper. It comprises the following:

1. Train a language-conditioned referring image segmentation model on the official OCID-VLG `unique` split;
2. Generate candidates through three planar 4-DoF grasping routes: end-to-end CROG, modular HiFi-CS→GR-ConvNet (G1), and HiFi-CS→GG-CNN2 (C1);
3. Place the frozen Top-5 candidates from each route under one evaluation contract;
4. Train, select, and gate candidate re-rankers using Train/Validation labels only;
5. After locking the code, models, candidate pools, thresholds, and evaluation plan, execute the formal Test evaluation exactly once across the 7,675 Test samples;
6. Independently recompute the results and generate dissertation tables, figures, reports, and the final PDF.

The formal primary panel is **offline 4-DoF candidate re-ranking for CROG/G1/C1**. D1 (HiFi-CS→Dex-Net/GQ-CNN) is a separate retrospective/compatibility panel. GraspNet, VGN, AnyGrasp, VL-Grasp, LAVT, SAM/Florence, and related components are secondary prototypes, diagnostics, or future 6-DoF extensions; they must not be reported as part of the formal primary results.

### 0.1 Reproducibility Levels

| Level | What it permits | Current workspace | Fresh clone of the public repository |
| --- | --- | --- | --- |
| R0 source audit | Read the code, tests, dissertation, and contracts | Possible | Possible |
| R1 evidence rebuild | Rebuild tables, figures, and the dissertation from frozen formal outputs | Possible | **Not possible**: formal `runs/` are not committed |
| R2 computational reproduction | Retrain models, generate candidates, re-rank, and run the formal Test | **Partly possible**: the frozen primary-panel sources can replay unified re-ranking, and HiFi/CROG can be retrained independently; the G1/C1 initialisation ledger and exact D1 dependencies are incomplete | **Not fully possible**: data, weights, frozen manifests, and several source snapshots have not been published |
| R3 independent/near-bitwise reproduction | Obtain identical inputs by hash and recompute the results | Retained local assets can be verified | Not currently possible; the assets listed in Section 17 must first be published |

This guide therefore provides two routes:

- **Route A: the current local workspace.** Completed experiments can be audited, and the existing frozen G1/C1 sources can be used to rerun unified re-ranking in a new run. HiFi and CROG can be retrained independently. The full G1/C1 initialisation ledger from an empty directory is missing, and the old single-FiLM/assets for the exact D1 Docker image are also incomplete. These routes can only establish new experiments under their existing contracts; they cannot be described as command-for-command replays of the original runs.
- **Route B: a fresh public clone.** Environment setup, data acquisition, upstream source retrieval, unit tests, and secondary demos can be completed. Until the missing assets are supplied, it is not valid to claim reproduction of the dissertation's formal numerical results.

## 1. Research Question, Inputs, Outputs, and Success Criteria

### 1.1 Inputs and Outputs

One OCID-VLG sample contains an RGB image, depth image, natural-language referring expression, target box/mask, and one or more ground-truth grasp rectangles. See the official [OCID-VLG](https://github.com/gtziafas/OCID-VLG) description.

The grasp candidate used in this dissertation is a planar rectangle:

```text
g = (center_x, center_y, width, height, angle)
```

It is a 4-DoF/grasp-rectangle proxy in the camera image plane. It does not contain full three-dimensional translation, three-dimensional rotation, robot-arm inverse kinematics, collision-aware trajectories, or real closed-loop execution.

### 1.2 The Sole Authoritative Formal Evaluation Contract

A prediction succeeds if and only if **one and the same** ground-truth grasp rectangle satisfies both conditions:

```text
rotated IoU > 0.25
AND
180°-periodic angular error <= 30°
```

Important details:

- IoU is strictly greater than `0.25`, not greater than or equal to it;
- the angle is evaluated under the gripper's 180° symmetry;
- both conditions must be satisfied by the same GT rectangle; GT-A cannot satisfy IoU while GT-B satisfies the angle criterion;
- empty candidate sets and empty predictions remain in the denominator;
- before completion of the formal lock, Test labels must not be used for feature, threshold, model, or gate selection.

### 1.3 Formal Results to Reproduce

The table below comes from the current locked evidence. It is not a promise that a fresh run will produce bit-identical floating-point values. Hardware, library versions, and non-deterministic kernels can introduce differences, but the contract, sample identity, and directional conclusions must agree.

| Route | Scope | N | Native J@1 | J@1 after conservative gated re-ranking | Gain (percentage points) | Recovered/Harmful |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CROG | FORMAL_PRIMARY | 7,675 | 0.892248 | 0.923648 | +3.140 | 278 / 37 |
| G1 | FORMAL_PRIMARY | 7,675 | 0.475179 | 0.566384 | +9.121 | 729 / 29 |
| C1 | FORMAL_PRIMARY | 7,675 | 0.438176 | 0.562476 | +12.430 | 1,006 / 52 |
| D1 | D1_RETROSPECTIVE, reported separately | 7,675 | 0.328990 | 0.515179 | +18.619 | 1,596 / 167 |

“Recovered” means that native Top-1 failed while the gated selection succeeded. “Harmful” means that native Top-1 succeeded while the gated selection failed. The formal primary conclusion is that, when a candidate pool contains exploitable Top-K headroom, a conservative selector can improve Top-1. This does not establish that an offline rectangle metric is equivalent to physical grasp success.

### 1.4 Historical Results That Must Not Be Combined

The repository also contains older experiments, including historical CROG results on a `multiple` Test set of 17,749 samples and early G1/C1 results with different angle-sign/candidate contracts. These are `LEGACY_INCOMPATIBLE` and must not be merged with `FORMAL_PRIMARY`. The final interpretation is controlled by:

- `runs/four_route_evidence_consolidation_20260813T155455Z/03_tables/table_01_scope_contract_map.md`
- `runs/four_route_evidence_consolidation_20260813T155455Z/03_tables/table_04_native_vs_gated.md`
- `runs/fair_unified_reranking_20260809_103012/08_lock/FORMAL_TEST_LOCK.json`
- `runs/fair_unified_reranking_20260809_103012/15_independent_recompute/INDEPENDENT_RECOMPUTE.json`

## 2. End-to-End Workflow from Data to Dissertation

```text
OCID-VLG unique train/val/test
          │
          ├── HiFi-CS repeated-FiLM ── target mask ──┬── GR-ConvNet fine-tune ── G1 candidates
          │                                           └── GG-CNN2 fine-tune ──── C1 candidates
          │
          └── CROG end-to-end ──────────────────────────────────────────────── CROG candidates

CROG / G1 / C1 route-local frozen Top-5
          │
          ├── P0–P1: source audit, paired manifests, fair candidate freeze
          ├── P2–P5: development labels, grouped folds, feature extraction
          ├── P6–P10: calibration, model matrix, validation selection, gate/router, attribution
          ├── P11: pre-lock assembly + immutable formal lock
          ├── P12: exactly-once Test evaluation
          └── P15/post-formal: independent recompute, statistics, cases, figures
                                      │
                                      └── evidence bundle → LuaLaTeX thesis PDF

D1: HiFi-CS → Dex-Net candidates → GQ-CNN score → separate retrospective panel
```

## 3. Repository Map: The Role of Each Directory

| Path | Purpose | Part of the formal primary path? |
| --- | --- | --- |
| `MSc_Dissertation_Template_the_University_of_Manchester_EEE__2025_onwards/` | Dissertation LaTeX source | Yes, final writing |
| `src/unified_reranking/`, `tools/unified_reranking/` | Fair candidate generation, features, models, gates, locks, formal evaluation, and independent recomputation | **Yes, core** |
| `tests/unified_reranking/` | Formal-contract and lifecycle tests | Yes |
| `crog_reproduction/CROG/` | CROG source snapshot, Mac/MPS adaptation, training, evaluation, and candidate export | Yes, CROG route |
| `HiFi_reproduction/hifics/` | Pinned HiFi-CS upstream commit and local repeated-FiLM modifications | Yes, but the entire directory is ignored by Git |
| `HiFi_reproduction/tools/grasp4dof/` | Data contracts, G1/C1 transfer/fine-tuning, candidates, locks, and reports | Yes, modular route |
| `HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py` | D1 initialisation, mask export, and Dex-Net/GQ-CNN orchestration | D1 only, reported separately |
| `src/d1_reranking/`, `tools/d1_reranking/` | D1 retrospective candidate re-ranking | D1 only, reported separately |
| `src/gtmask_counterfactual/` | GT-mask counterfactual diagnostics | No, post-formal mechanism analysis |
| `target_aware_vlm_grasping/` | Florence-2/SAM plus CPU geometric-candidate demo | No, secondary prototype/quick start |
| `src/graspnet6d/`, `legacy/external_graspnet/`, and VGN/AnyGrasp-related directories | 6-DoF subset and extension experiments | No; they cannot replace the formal 4-DoF primary results |
| `LAVT_reproduction/` | GPL-3.0 referring-segmentation baseline | No, optional |
| `runs/`, `HiFi_reproduction/runs/` | Locked run evidence, candidates, models, figures, and reports | Yes locally; absent from the public repository |

## 4. Capability and Environment Audit

### 4.1 Capabilities Verified on the Current Machine

The audited workspace has an Apple M5 Pro with 24 GiB unified memory, PyTorch MPS available, and no CUDA. The Docker CLI is installed, but its daemon was not running. ROS, MoveIt, and Gazebo are absent. Consequently:

- MPS/CPU can rerun HiFi-CS, G1, C1, the adapted CROG route, and unified re-ranking;
- the TensorFlow 1.15/Python 3.7 GQ-CNN route should use a `linux/amd64` Docker container rather than the current native Python;
- this environment can validate offline metrics only; it cannot execute robot grasps;
- disk space is already constrained, so run `df -h` and `du -sh` before downloading duplicate data or weights.

The following isolated environments already exist and have been verified:

```text
HiFi_reproduction/.venv-grasp4dof   Python 3.11, Torch/MPS, primary 4-DoF and unified re-ranking
HiFi_reproduction/.venv-gqcnn       candidate-side dependencies; not a native TF1.15 runtime
.venv-graspnet6d                     optional 6-DoF subset tools
.venv-paper-tools                    plotting/dissertation support; currently lacks pyarrow and cannot read evidence Parquet directly
```

Do not force all modules into one environment. CROG, HiFi-CS, 4-DoF, GQ-CNN, and dissertation construction come from different dependency generations.

### 4.2 Capabilities Used and Not Used

- Used: repository tests, `commands.log` ledgers, locked JSON/Parquet evidence, SHA-256, Git, existing virtual environments, and official GitHub/paper/model pages.
- Optional: Docker (D1/GQ-CNN only), `gdown` (Google Drive downloads), and LuaLaTeX/`latexmk` (dissertation construction).
- Not required: ROS/MoveIt/Gazebo, online/physical robot execution, or CUDA, unless switching to the official CROG CUDA path or a physical extension.
- Excluded from formal reproduction: the Florence/SAM demo, GraspNet/VGN/AnyGrasp, LAVT, and generic plotting/data-analysis plugins. They do not control the dissertation's primary numbers.

## 5. Create a Clean, Auditable Working Directory

### 5.1 Fresh Clone

```bash
git clone https://github.com/langchengg/VLMGraspPose.git
cd VLMGraspPose
git checkout 461505999123fe23303bab45bfc7ddbf9619f1fb

export VLMGP_ROOT="$(pwd)"
git status --short
git rev-parse HEAD
```

If working in the current local repository, run only:

```bash
cd /path/to/VLMGraspPose
export VLMGP_ROOT="$(pwd)"
```

The current worktree already contains the user's own deletion state. Do not run `git reset --hard`, `git clean -fdx`, or any command that could erase data, run evidence, or user changes.

### 5.2 Create a New Run ID

Every rerun must write to a new directory. Never overwrite `runs/fair_unified_reranking_20260809_103012` or another locked source run:

```bash
export REPRO_ID="repro_$(date -u +%Y%m%dT%H%M%SZ)"
export REPRO_ROOT="${VLMGP_ROOT}/runs/${REPRO_ID}"
mkdir -p "${REPRO_ROOT}"
printf '%s\n' "${REPRO_ROOT}"
```

Record the host and Git state:

```bash
uname -a > "${REPRO_ROOT}/host.txt"
git rev-parse HEAD > "${REPRO_ROOT}/git_head.txt"
git status --porcelain=v1 > "${REPRO_ROOT}/git_status.txt"
python3 --version > "${REPRO_ROOT}/python.txt"
```

## 6. Download, Verify, and Place the OCID-VLG Dataset

### 6.1 Official Source and Licensing Boundary

The official dataset page is [gtziafas/OCID-VLG](https://github.com/gtziafas/OCID-VLG), whose download link points to the authors' [Google Drive file](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link). The official description reports 89,639 image-text-mask-grasp tuples from 1,763 scenes and provides four splits: `multiple`, `unique`, `novel-instances`, and `novel-classes`.

The OCID-VLG repository does not provide a clear dataset-level redistribution licence, and no official SHA-256 is published for the Drive archive. By default, use it locally for academic research with proper citation. Do not commit or redistribute the dataset archive through this repository.

### 6.2 Download

Prefer opening the Drive link above in a browser. For command-line use, install `gdown` in an isolated download environment:

```bash
python3 -m venv "${VLMGP_ROOT}/.venv-download"
"${VLMGP_ROOT}/.venv-download/bin/python" -m pip install --upgrade pip gdown
mkdir -p "${VLMGP_ROOT}/downloads/ocid_vlg"

"${VLMGP_ROOT}/.venv-download/bin/gdown" \
  --fuzzy "https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link" \
  -O "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip"
```

If Google Drive returns a quota or confirmation-page error, stop retrying and download through a browser. Do not use an unverified mirror.

### 6.3 Post-download Integrity Checks

Because the authors do not publish an archive hash, record a local hash after the first trusted download:

```bash
file "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip"
unzip -t "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip"
shasum -a 256 "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip" \
  | tee "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip.sha256"
```

Do not mistake the following frozen-manifest hashes for an official ZIP hash. They verify only the sample manifests generated for this dissertation:

| `unique` manifest | Samples | SHA-256 |
| --- | ---: | --- |
| Train | 26,295 | `a986bcce3e1961be816a295c3ae0942e64e61275524a85c0a8957563e7f920c1` |
| Validation | 3,778 | `573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a` |
| Test | 7,675 | `915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409` |
| Three-manifest set | 37,748 | `fa36db4ff548f2ca06abadefb30cacc741165478dff8691203375a311b86c0c8` |

### 6.4 Extraction and Directory Structure

```bash
mkdir -p "${VLMGP_ROOT}/datasets/OCID-VLG"
unzip "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip" \
  -d "${VLMGP_ROOT}/datasets/OCID-VLG"

find "${VLMGP_ROOT}/datasets/OCID-VLG" -maxdepth 2 -type d | sort | sed -n '1,80p'
```

If the archive contains an extra top-level directory, define the actual `OCID-VLG` root as `OCIDVLG_ROOT`. At minimum, that root should contain:

```text
OCID-VLG/
├── ARID10/
├── ARID20/
├── refer/
│   ├── multiple/
│   ├── unique/
│   ├── novel-instances/
│   └── novel-classes/
└── catalog.csv (if included in the downloaded version)
```

Use one source of truth and symbolic links for the consuming subprojects:

```bash
export OCIDVLG_ROOT="${VLMGP_ROOT}/datasets/OCID-VLG"

ln -s "${OCIDVLG_ROOT}" "${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG"
ln -s "${OCIDVLG_ROOT}" "${VLMGP_ROOT}/crog_reproduction/OCID-VLG"
ln -s "${OCIDVLG_ROOT}" "${VLMGP_ROOT}/target_aware_vlm_grasping/data/OCID-VLG"
```

If a target path already exists, inspect it with `readlink`/`ls -ld` first. Do not overwrite it with `ln -sf`. The CROG data-root setting must also point to the same `OCIDVLG_ROOT`, avoiding drift between three copies.

### 6.5 Dataset Manifest Verification

The current local workspace retains the frozen manifests:

```bash
cd "${VLMGP_ROOT}"
shasum -a 256 \
  HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_train.json \
  HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_val.json \
  HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_test.json
```

These manifests are ignored in a fresh public clone. They must be restored from a published reproduction asset package or rebuilt with the repository tools and treated as a new experiment contract. A newly generated set must not be represented as having exactly the same identity as the table above.

## 7. Obtain Pinned Upstream Source Code and Model Weights

### 7.1 Third-party Source Code

Read `docs/THIRD_PARTY_NOTICES.md` before running the pinned-commit retrieval script:

```bash
cd "${VLMGP_ROOT}"
sed -n '1,260p' docs/THIRD_PARTY_NOTICES.md
bash scripts/fetch_external_repositories.sh
```

The table combines pinned source snapshots already present in the repository with dependency identities needed by a fresh clone. `fetch_external_repositories.sh` clones only HiFi-CS, GQ-CNN, the GraspNet API/baseline, and VL-Grasp. CROG, GR-ConvNet, GG-CNN, LAVT, and VGN are repository snapshots; OpenAI CLIP is not fetched by this script, and its pin is recorded separately in `configs/graspnet6d/environment.lock.txt` and frozen environment evidence. The script uses the network and writes missing directories, but does not overwrite existing ones. Important pinned commits are:

| Component | Pinned commit | Licence/use boundary |
| --- | --- | --- |
| CROG | `1eeee85de1fe6bffdc66c9ed9a622028ea04578e` | MIT; source includes the local adaptation |
| OpenAI CLIP | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | Source code is MIT; the pretrained weights have no separate, clear weight licence statement |
| HiFi-CS | `4be6b3be7ce79fae481fb51616adfa2b803f07a0` | No repository-level licence found during the audit; local research/reference only, no redistribution |
| GR-ConvNet | `bdd49367f8619be94123fb3187c2f8ad5100ef46` | BSD-3-Clause |
| GG-CNN | `0c50aa7600e8a30d44c5c85cebd6e3394a81f30e` | BSD-3-Clause |
| GQ-CNN | `499a609fe9dfb074bdfb6c4e6e33667ea50f4c21` | Custom educational, research, and not-for-profit terms |
| LAVT-RIS | `1da0af9f21b637c0cae9ea1363d2dd9b40e19628` | GPL-3.0, isolated subtree |
| VGN | `d7af0622433f52ae88ebe81533f12b46b33e951a` | BSD-3-Clause; optional 6-DoF route |

Verify the retrieved sources:

```bash
git -C HiFi_reproduction/hifics rev-parse HEAD
git -C HiFi_reproduction/third_party/gqcnn-official rev-parse HEAD
```

### 7.2 Weight Sources, Paths, and Hashes

Download weights only from official release locations. PyTorch `.pth/.pt/.bin` files commonly use pickle and must be treated as untrusted executable inputs. Verify the hash before loading them in an isolated environment.

| Weight | Official source | Dissertation-expected SHA-256/notes |
| --- | --- | --- |
| OpenAI CLIP RN50 | [RN50.pt](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) | `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762` |
| OpenAI CLIP ViT-B/16 | [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| GR-ConvNet Cornell | [Official file at the pinned commit](https://github.com/skumra/robotic-grasping/blob/bdd49367f8619be94123fb3187c2f8ad5100ef46/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98) | 7,640,481 bytes; formal initialisation hash `e6c03afc0f8266d29ec92141088cc6a557215fabc2681d4ca5a162f451853d28` |
| GG-CNN2 Cornell | [v0.1 release ZIP](https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn2_weights_cornell.zip) | ZIP `f71e3575fe70bea6817239f9fad98264af5cfd95dd9bbe69ff4144696f34f972`; extracted state dict `865d538a51d427f7ee84defc99e093bdf51eeb0627c302068037e52188c11d1c` |
| GQ-CNN 2.1 | Model download is handled by the repository script | model-zoo archive `c3823f3525df851ea0b75c202e96e131ed85bafa9d4f3c4daa270a3c80943472`; `config.json` `eb5bc17089a39bd8fe6c801010c25a6a79a898d64181180feb5cf69aa630ff6f`; canonical 22-file manifest `8201961abe3a09d90c6c66e582a3bfeb181d7095a2ebcc3a9d90e68fc12e8614` |

The BSD-3-Clause findings for GR-ConvNet and GG-CNN apply to their upstream source code. The upstream projects do not provide a separate, clear licence for the pretrained weights. The weights can be downloaded for local reproduction, but redistribution rights should be confirmed separately. The GQ-CNN downloader first attempts the historical script in the pinned official repository. If the old Box links fail, it falls back to the Drive model zoo linked by the official Dex-Net documentation.

Download CLIP and verify it strictly:

```bash
mkdir -p "${VLMGP_ROOT}/models/clip"

curl -fL --retry 3 \
  "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt" \
  -o "${VLMGP_ROOT}/models/clip/RN50.pt"

curl -fL --retry 3 \
  "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt" \
  -o "${VLMGP_ROOT}/models/clip/ViT-B-16.pt"

printf '%s  %s\n' \
  afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762 \
  "${VLMGP_ROOT}/models/clip/RN50.pt" | shasum -a 256 -c -

printf '%s  %s\n' \
  5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f \
  "${VLMGP_ROOT}/models/clip/ViT-B-16.pt" | shasum -a 256 -c -
```

The download directory is only a common storage location. The two loaders read different fixed paths. After confirming that the destinations do not exist, create links:

```bash
mkdir -p "${VLMGP_ROOT}/crog_reproduction/CROG/exp/pretrain_clip"
mkdir -p "${HOME}/.cache/clip"

ln -s "${VLMGP_ROOT}/models/clip/RN50.pt" \
  "${VLMGP_ROOT}/crog_reproduction/CROG/exp/pretrain_clip/RN50.pt"
ln -s "${VLMGP_ROOT}/models/clip/ViT-B-16.pt" \
  "${HOME}/.cache/clip/ViT-B-16.pt"

shasum -a 256 \
  "${VLMGP_ROOT}/crog_reproduction/CROG/exp/pretrain_clip/RN50.pt" \
  "${HOME}/.cache/clip/ViT-B-16.pt"
```

The CROG Mac configuration reads `exp/pretrain_clip/RN50.pt`; the HiFi trainer reads `~/.cache/clip/ViT-B-16.pt`. If a destination already exists, verify its hash rather than overwriting it.

The formal GR-ConvNet initialisation file belongs at:

```text
HiFi_reproduction/third_party_src/grconvnet/trained-models/
└── cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98
```

This dissertation uses the Cornell weight, not a Jacquard weight that may be present in the same directory. The GG-CNN2 state dict belongs at:

```text
HiFi_reproduction/third_party_src/checkpoints/ggcnn2/
└── ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt
```

Download GG-CNN2 from the official release:

```bash
export GGCNN_DIR="${VLMGP_ROOT}/HiFi_reproduction/third_party_src/checkpoints/ggcnn2"
mkdir -p "${GGCNN_DIR}"
curl -fL --retry 3 \
  "https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn2_weights_cornell.zip" \
  -o "${GGCNN_DIR}/ggcnn2_weights_cornell.zip"

printf '%s  %s\n' \
  f71e3575fe70bea6817239f9fad98264af5cfd95dd9bbe69ff4144696f34f972 \
  "${GGCNN_DIR}/ggcnn2_weights_cornell.zip" | shasum -a 256 -c -

unzip -q "${GGCNN_DIR}/ggcnn2_weights_cornell.zip" -d "${GGCNN_DIR}"
printf '%s  %s\n' \
  865d538a51d427f7ee84defc99e093bdf51eeb0627c302068037e52188c11d1c \
  "${GGCNN_DIR}/ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt" \
  | shasum -a 256 -c -
```

The pinned upstream GR-ConvNet commit directly tracks the 7,640,481-byte Cornell weight. Use the commit-pinned Raw URL, not a moving `main` branch or a third-party mirror:

```bash
export GR_CKPT="${VLMGP_ROOT}/HiFi_reproduction/third_party_src/grconvnet/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
mkdir -p "$(dirname "${GR_CKPT}")"
curl -fL --retry 3 \
  "https://raw.githubusercontent.com/skumra/robotic-grasping/bdd49367f8619be94123fb3187c2f8ad5100ef46/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98" \
  -o "${GR_CKPT}"
printf '%s  %s\n' \
  e6c03afc0f8266d29ec92141088cc6a557215fabc2681d4ca5a162f451853d28 \
  "${GR_CKPT}" | shasum -a 256 -c -
```

This download restores the formal G1/G0 initialisation-weight identity. However, the complete initialisation command ledger for the original G1/C1 runs from an empty directory was not retained. A new training run must therefore be described as an independent experiment under the current tool contract, not a command-for-command replay of the original run.

Hashes of the locally fine-tuned formal weights:

```text
G1 best_state_dict.pt  5fc8cdae2578a2361c80d19d44ebf0df521938cd4c00c04fedb986b1f5169aea
C1 best_state_dict.pt  13addaa29f1f108888946b50467731b6fb53e34d4dcea5ba9d1d741a95147bc9
```

These fine-tuned weights and the formal CROG weight are not tracked by Git. A fresh clone must either obtain and verify them from a published asset package or retrain them as described below.

## 8. Create Isolated Environments for Each Route

### 8.1 Main 4-DoF/Unified Re-ranking Environment

Python 3.11 is recommended for macOS/MPS:

```bash
cd "${VLMGP_ROOT}"
uv venv --python 3.11 HiFi_reproduction/.venv-grasp4dof
uv pip install --python HiFi_reproduction/.venv-grasp4dof/bin/python \
  -r HiFi_reproduction/requirements-grasp4dof-macos.txt

export PY4="${VLMGP_ROOT}/HiFi_reproduction/.venv-grasp4dof/bin/python"
"${PY4}" -c 'import torch; print(torch.__version__); print("MPS", torch.backends.mps.is_available())'
PYTHONPATH=src "${PY4}" -m pytest -q tests/unified_reranking/test_metrics.py
```

If `uv` is unavailable, create the environment with `python3.11 -m venv` and run `pip install -r ...` inside it. Do not upgrade dependencies without a reason; retain the formal run's `package_lock.txt` as the preferred comparison point.

### 8.2 HiFi-CS Repeated-FiLM Environment

Upstream HiFi-CS does not provide a stable lockfile. The dissertation run depends on local modifications and a runtime source snapshot. First check the current machine:

```bash
test -f HiFi_reproduction/hifics/tools/train_hierfilm.py
test -f HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml
test -f HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_train.json
```

If any check fails, a public clone does not have the prerequisites for exact training. Do not guess missing files from upstream `main`; restore the source snapshot, configuration, and manifests listed in Section 17. Create the environment as follows:

```bash
python3.11 -m venv "${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics"
"${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics/bin/python" -m pip install --upgrade pip
"${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics/bin/python" -m pip install \
  -r "${VLMGP_ROOT}/HiFi_reproduction/hifics/requirements-macos.txt"
```

### 8.3 CROG Environment

For the official CUDA route, use CROG's `environment.yml`. For this repository's macOS adaptation, use `README_MAC.md` and `requirements_mac.txt`:

```bash
cd "${VLMGP_ROOT}/crog_reproduction/CROG"
python3.11 -m venv .venv-crog
.venv-crog/bin/python -m pip install --upgrade pip
.venv-crog/bin/python -m pip install -r requirements_mac.txt
.venv-crog/bin/python scripts/check_mps.py
```

If the official dependencies require a different Python version, create a separate environment rather than merging them into `.venv-grasp4dof`.

### 8.4 GQ-CNN/D1 Docker Environment

Upstream GQ-CNN v1.3.0 supports Python 3.5–3.7 and TensorFlow `<=1.15.0`; this repository pins Python 3.7.17 and TensorFlow 1.15.0 in its isolated scoring container. The repository supplies a pinned Dockerfile:

```bash
cd "${VLMGP_ROOT}"
docker info
docker build --platform linux/amd64 \
  -t vlmgrasp/gqcnn-score:1.3.0 \
  -f HiFi_reproduction/docker/gqcnn-score/Dockerfile \
  HiFi_reproduction
```

If `docker info` fails, the daemon is not running. Start Docker Desktop before continuing. Do not give an unverified image a writable mount of the repository. During formal scoring, mount repository source and inputs read-only, and expose only the model directory as writable where the script explicitly requires it.

## 9. Run Minimal Smoke Tests Before Training for Hours

### 9.1 Secondary Target-aware Demo

This demo validates data loading, target masks, RGB-D geometric candidates, re-ranking, and the output format, but it is **not a formal dissertation model**.

```bash
cd "${VLMGP_ROOT}/target_aware_vlm_grasping"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python scripts/run_one_sample.py \
  --dataset ocid_vlg \
  --dataset-root data/OCID-VLG \
  --index 0 \
  --target-source oracle \
  --scorer rule_based \
  --output-root outputs/repro_oracle_smoke \
  --top-k 5

.venv/bin/python scripts/evaluate_outputs.py \
  --output-root outputs/repro_oracle_smoke \
  --mode ocid_2d
```

Begin with `oracle` to isolate grasp-side problems. Only after it passes should `requirements-vlm.txt` be installed and Florence-2 plus SAM attempted. The VLM mode is a current demo and must not be entered in the formal results table.

### 9.2 Formal Toolchain Smoke Tests

```bash
cd "${VLMGP_ROOT}"
export PY4="${VLMGP_ROOT}/HiFi_reproduction/.venv-grasp4dof/bin/python"

"${PY4}" -m tools.unified_reranking.pipeline_status --help
"${PY4}" HiFi_reproduction/tools/grasp4dof/run_training_grid.py --help
PYTHONPATH=src "${PY4}" -m pytest -q \
  tests/unified_reranking/test_metrics.py \
  tests/unified_reranking/test_contracts.py \
  tests/unified_reranking/test_test_access_guard.py
PYTHONPATH=src "${PY4}" -m pytest -q \
  tests/unified_reranking/test_independent_recompute.py::test_exact_mcnemar_is_symmetric_and_handles_no_discordance
```

The checks above are a seconds-long smoke suite. Formal-lock lifecycle and independent-recomputation tests build larger synthetic tables and may take several minutes on this machine. Before starting the formal Test, also run:

```bash
PYTHONPATH=src "${PY4}" -m pytest -q \
  tests/unified_reranking/test_formal_test_orchestration.py::test_formal_lock_is_complete_immutable_and_hash_verified \
  tests/unified_reranking/test_formal_test_orchestration.py::test_formal_test_claims_before_single_label_read_and_finalizes_once
```

Proceed to full training only after the dataset identity, weight hashes, tests, and smoke runs all pass.

## 10. Stage A: Reproduce HiFi-CS Repeated-FiLM Referring Image Segmentation

### 10.1 Outputs of This Stage

HiFi-CS maps a referring expression and RGB image to a target mask. It does not generate a grasp pose. The dissertation uses a locally implemented five-level hierarchical repeated-FiLM variant, retrained under a controlled protocol on OCID-VLG `unique`. The run configuration states explicitly:

```text
controlled official-unique retraining; not an exact Table 2 reproduction
```

This stage therefore reproduces the segmentation front end used by the dissertation, not every entry of Table 2 in the HiFi-CS paper.

### 10.2 Verify the Frozen Training Contract

Source run directory:

```text
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/
```

Inspect its evidence before starting any training:

```bash
cd "${VLMGP_ROOT}"
export HIFI_SOURCE="${VLMGP_ROOT}/HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615"

test -f "${HIFI_SOURCE}/COMPLETE"
jq '{mode, device, config_sha256, source_snapshot_sha256, git, manifests}' \
  "${HIFI_SOURCE}/startup_evidence.json"

# Verify against the per-file manifest emitted by the trainer. Do not run
# shasum directly on source_snapshot/* because it may contain __pycache__/.
(
  cd "${HIFI_SOURCE}/source_snapshot"
  jq -r 'to_entries[] | "\(.value)  \(.key)"' \
    ../source_snapshot_sha256.json | shasum -a 256 -c -
)

export HIFI_SNAPSHOT_AGGREGATE="$(
  jq -cS . "${HIFI_SOURCE}/source_snapshot_sha256.json" \
    | tr -d '\n' | shasum -a 256 | awk '{print $1}'
)"
test "${HIFI_SNAPSHOT_AGGREGATE}" = \
  "$(cat "${HIFI_SOURCE}/source_snapshot_aggregate_sha256.txt")"
test "${HIFI_SNAPSHOT_AGGREGATE}" = \
  "$(jq -r .source_snapshot_sha256 "${HIFI_SOURCE}/startup_evidence.json")"
```

Key contract values:

| Item | Value |
| --- | --- |
| Input resolution | 352×352 |
| CLIP | ViT-B/16; image and text encoders frozen |
| Repeated-FiLM levels | 5 levels; projection layers 1/3/5/7/9; decoding order 9/7/5/3/1 |
| Optimiser | Adam, LR 0.001, weight decay 0 |
| Batch / accumulation | 16 / 1 |
| Steps | Maximum 20,000 optimiser steps |
| Scheduler | Cosine, `T_max=20000`, `eta_min=0.0001` |
| Seed / dtype | 42 / float32 |
| Formal local device | MPS, AMP disabled, workers 0 |
| Selection rule | Maximum Validation foreground mIoU |
| Test use | Once, after freezing the best Validation checkpoint |

Some hyperparameters were not disclosed in the HiFi-CS paper. They are documented released-code/controlled-run choices and must not be described as requirements of the original paper.

### 10.3 Configure a New Run

The current machine should contain:

```text
HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml
```

Copy it to a dedicated configuration and change only absolute paths and the new output directory, not algorithmic fields:

```bash
export HIFI_ID="hifics_ocidvlg_hierfilm_repro_$(date -u +%Y%m%dT%H%M%SZ)"
export HIFI_RUN="${VLMGP_ROOT}/HiFi_reproduction/runs/${HIFI_ID}"
export HIFI_CONFIG="${VLMGP_ROOT}/HiFi_reproduction/configs/${HIFI_ID}.yaml"

cp "${VLMGP_ROOT}/HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml" \
  "${HIFI_CONFIG}"
```

Confirm in `HIFI_CONFIG` that:

- `dataset_root` points to `${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG`;
- the train/val/test manifests point to the frozen manifests;
- `output_directory_pattern`, or CLI `--run-dir`, points to the new `HIFI_RUN`;
- `resume: false` and `checkpoint: null`;
- Test labels and Test evaluation files remain unopened before the first formal run.

Record the configuration hash:

```bash
shasum -a 256 "${HIFI_CONFIG}"
```

### 10.4 Overfitting/Small-sample Check

The trainer supports only `full` and `overfit`. Run a constrained overfit check first and never treat it as a formal result:

```bash
cd "${VLMGP_ROOT}/HiFi_reproduction/hifics"
export PYH="${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics/bin/python"

"${PYH}" tools/train_hierfilm.py "${HIFI_CONFIG}" \
  --mode overfit \
  --run-dir "${HIFI_RUN}_overfit" \
  --max-steps 200 \
  --limit-samples 32
```

The overfit gate requires exactly 32 samples and 200–500 optimiser steps; other values are rejected. Confirm that the loss is finite, parameters change, the checkpoint reloads, and mask semantics are not inverted. The formal source run retains `parameter_max_abs_changes.json`, `optimizer_coverage.json`, and qualitative audits for comparison.

### 10.5 Full Training and Independent Verification

```bash
cd "${VLMGP_ROOT}/HiFi_reproduction/hifics"

"${PYH}" tools/train_hierfilm.py "${HIFI_CONFIG}" \
  --mode full \
  --run-dir "${HIFI_RUN}"

"${PYH}" tools/verify_hierfilm_training.py "${HIFI_RUN}"
"${PYH}" tools/evaluate_hierfilm.py "${HIFI_RUN}" --device mps
"${PYH}" tools/verify_hierfilm_evaluation.py "${HIFI_RUN}"
```

`train_hierfilm.py` performs training and Validation only; it explicitly does not load Test during the training loop. The final three commands generate `evaluation/` and the two independent verification files. Once `verify_hierfilm_evaluation.py` passes, the new repeated-FiLM run has completed its own training, one route-local HiFi Test evaluation, and independent verification.

`report_hierfilm_results.py` is **not** a self-contained finaliser for a new run. It also reads a structural test and the old single-FiLM baseline. Run it only when all three prerequisite assets below are present and the historical comparison report is genuinely required:

```text
HiFi_reproduction/artifacts/experiment/hierarchical_film_tests.json
HiFi_reproduction/runs/hifics_ocidvlg_20260711_112921/evaluation/evaluation_metrics.json
HiFi_reproduction/reports/hifics_previous_checkpoint_recomputed_protocols.json
```

The first can be generated from the current source. The latter two require restoration of the old single-FiLM run; the third is generated by re-evaluating the old checkpoint:

```bash
"${PYH}" tools/verify_hierarchical_film.py \
  --device mps \
  --train-manifest "${VLMGP_ROOT}/HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_train.json" \
  --json-output "${VLMGP_ROOT}/HiFi_reproduction/artifacts/experiment/hierarchical_film_tests.json" \
  --report-output "${VLMGP_ROOT}/HiFi_reproduction/reports/hierarchical_film_tests.md"

test -f "${VLMGP_ROOT}/HiFi_reproduction/runs/hifics_ocidvlg_20260711_112921/evaluation/evaluation_metrics.json"
if ! test -f "${VLMGP_ROOT}/HiFi_reproduction/reports/hifics_previous_checkpoint_recomputed_protocols.json"; then
  "${PYH}" tools/evaluate_previous_fixed_protocols.py \
    --previous-run "${VLMGP_ROOT}/HiFi_reproduction/runs/hifics_ocidvlg_20260711_112921" \
    --test-manifest "${VLMGP_ROOT}/HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_test.json" \
    --output "${VLMGP_ROOT}/HiFi_reproduction/reports/hifics_previous_checkpoint_recomputed_protocols.json" \
    --device mps
fi

"${PYH}" tools/report_hierfilm_results.py "${HIFI_RUN}"
```

`evaluate_previous_fixed_protocols.py` refuses to overwrite an existing output. The audited workspace currently lacks both the old run's `evaluation_metrics.json` and its prior recomputation file, so the **historical baseline comparison report** cannot be rebuilt directly from the current state. This does not block completion of the new repeated-FiLM run itself.

The original command for the formal source run is recorded in:

```text
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/command.json
```

Completion criteria:

```bash
test -f "${HIFI_RUN}/COMPLETE"
test -f "${HIFI_RUN}/evaluation/EVALUATION_COMPLETE.json"
jq . "${HIFI_RUN}/run_summary.json"
jq . "${HIFI_RUN}/evaluation/independent_metric_verification.json"
shasum -a 256 "${HIFI_RUN}/checkpoints/best.pth"
```

The source run's best checkpoint was selected at epoch 12 / step 19,728, with Validation foreground mIoU of approximately 0.820741. Its SHA-256 is:

```text
b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601
```

A new independent run need not produce an identical hash, but it must retain the configuration, source snapshot, manifests, device, random seed, training history, best-checkpoint selection rule, and independent recomputation.

## 11. Stage B: Reproduce the G1/C1 Modular 4-DoF Back Ends

### 11.1 Route Definitions

- `R0`: repeated-FiLM segmentation reference, not a grasp back end;
- `G0`: GR-ConvNet pretrained transfer;
- `G1`: GR-ConvNet fine-tuned on OCID-VLG;
- `C0`: GG-CNN2 pretrained transfer;
- `C1`: GG-CNN2 fine-tuned on OCID-VLG;
- `A0`: analytical/geometric baseline;
- `*-O`: oracle-mask diagnostic used only to localise segmentation bottlenecks.

The unified fair primary panel ultimately uses G1 and C1. The formal source run is:

```text
HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500
```

### 11.2 Source and Input Audit

```bash
cd "${VLMGP_ROOT}"
export PY4="${VLMGP_ROOT}/HiFi_reproduction/.venv-grasp4dof/bin/python"
export MOD_SOURCE="${VLMGP_ROOT}/HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500"

test -f "${MOD_SOURCE}/FINALIZATION_COMPLETE.json"
jq . "${MOD_SOURCE}/audit/source_inventory.json"
jq . "${MOD_SOURCE}/audit/dataset_split_audit.json"
jq . "${MOD_SOURCE}/audit/formal_input_reference_preflight.json"
jq . "${MOD_SOURCE}/selected_configs/G1.json"
jq . "${MOD_SOURCE}/selected_configs/C1.json"
```

An important operational boundary applies here: `audit_sources.py` hard-codes verification of the formal source HiFi run and checkpoint SHA, together with two compact input sources:

```text
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615
HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528
HiFi_reproduction/runs/modular_reranking_repeatedfilm_v1_20260729_203147
```

Consequently, the new `HIFI_RUN` trained in Section 10 is not consumed automatically by Section 11. Replaying the dissertation's locked pipeline requires the formal HiFi checkpoint above. Connecting a new HiFi checkpoint to G1/C1 requires a changed source contract, regenerated compact inputs, a new audit, and a new experiment. The formal dissertation hashes and result names cannot be reused.

### 11.3 Why a Fictitious “One-command Replay” Cannot Be Provided

This subsystem has no verified top-level entry point that can safely execute all stages from an empty directory. The source facts are:

- the stage tools are under `HiFi_reproduction/tools/grasp4dof/`;
- the corrected source run's `commands.log` contains only 20 entries, beginning with `recompute_reference` and the formal Test tail;
- the parent run `modular_repeatedfilm_4dof_backends_v1_20260803_092055/commands.log` contains only 11 entries covering the G1/C1 training grid, part of Validation, and selection;
- the original audit, manifest, transfer-tuning, and lock-creation commands do **not have a complete ledger**. The manifests and locks verify the outputs but cannot reconstruct every initialisation command;
- both `commands.log` files contain absolute source-machine paths and must not be executed line by line without review;
- the Test stage is already locked, and the old run must not be executed again or overwritten.

Render the two incomplete ledgers as readable lists first:

```bash
sed -n '1,240p' "${MOD_SOURCE}/commands.log"
sed -n '1,240p' \
  "${VLMGP_ROOT}/HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_20260803_092055/commands.log"
```

Create a new modular run directory and inspect the current CLI with `--help` at each step:

```bash
export MOD_ID="modular_repeatedfilm_4dof_repro_$(date -u +%Y%m%dT%H%M%SZ)"
export MOD_RUN="${VLMGP_ROOT}/HiFi_reproduction/runs/${MOD_ID}"
mkdir -p "${MOD_RUN}"

for tool in \
  audit_sources.py \
  build_manifests.py \
  select_development_subsets.py \
  tune_transfer_backends.py \
  run_training_grid.py \
  select_validation_primary.py \
  build_formal_lock_config.py \
  lock_experiment.py \
  run_method.py \
  consolidate_results.py \
  independent_recompute.py; do
  "${PY4}" "HiFi_reproduction/tools/grasp4dof/${tool}" --help
done
```

### 11.4 Intended Stage Order (Not a Verified One-command Replay)

The following order can be established from code dependencies and frozen manifests, but the complete original CLI for steps 1–4 was not retained. To establish a new independent experiment, inspect every current `--help`, use `--dry-run` first where supported, and record the actual commands in a new ledger. Do not describe this as a command-for-command replay of the original run:

1. `audit_sources.py`: pin HiFi, GR-ConvNet, GG-CNN2 source and pretrained-weight hashes;
2. `build_manifests.py`: construct Train 26,295 / Validation 3,778 / Test 7,675 manifests;
3. `select_development_subsets.py`: construct smoke/pilot subsets using Train/Validation only;
4. `tune_transfer_backends.py`: tune G0/C0 transfer parameters on Validation only;
5. `run_training_grid.py`: fine-tune G1/C1 on Train and select checkpoints with Validation;
6. `select_validation_primary.py`: freeze the primary method and configuration;
7. `build_formal_lock_config.py`: produce the candidate-lock configuration;
8. `lock_experiment.py`: lock the code, data, models, configuration, and evaluator;
9. run the formal Test only after the lock: `recompute_reference.py` generates R0 separately; `run_method.py --split test` supports only G0/G1/C0/C1/A0;
10. `consolidate_results.py`, statistics, and `independent_recompute.py`: consolidate and independently recompute;
11. generate reports/galleries and mark completion.

Before step 8, Test may be represented only by sample identities without labels. `test_labels.parquet` must not be read for configuration selection.

If the objective is to reproduce the dissertation's **core re-ranking conclusion** rather than retrain the grasp back ends, the most reliable current route is to treat the corrected source run as an immutable G1/C1 source and create a new unified run through Section 13. If the objective is complete from-scratch G1/C1 retraining, the correct current status is `PARTIALLY REPRODUCED` until an initialisation driver or complete command ledger is published.

### 11.5 Locked G1/C1 Selection Values

| Parameter | G1 | C1 |
| --- | ---: | ---: |
| Input size | 224 | 300 |
| Mask dilation ratio | 0.10 | 0.15 |
| Quality threshold | 0.10 | 0.05 |
| Minimum peak distance | 15 px | 20 px |
| Raw-candidate limit | 100 | 100 |
| Fixed rectangle height | 20 px | 20 px |
| Seed | 20260803 | 20260803 |
| LR / weight decay | 1e-4 / 1e-5 | 1e-4 / 0 |
| Maximum epochs / patience | 30 / 5 | 30 / 5 |
| Best epoch | 23 | 30 |

The machine-readable sources of truth are `selected_configs/G1.json` and `C1.json`; the table is only a reading aid.

### 11.6 Important Contract Mismatch

The approximately 0.87/0.79 G1/C1 values in the modular source report come from the earlier native candidate contract of the “corrected back-end experiment”. The unified fair run redefined and froze candidate decoding under one evaluation contract; its formal native values are 0.475179/0.438176. The two experiments answer different questions under different candidate contracts. Their difference must not be interpreted as either a code regression or a re-ranking effect.

## 12. Stage C: Reproduce the CROG Route

CROG is the official end-to-end CLIP-based referring grasp synthesis model. See the [PMLR paper page](https://proceedings.mlr.press/v229/tziafas23a.html) and [official repository](https://github.com/HilbertXu/CROG). The official work defines 4-DoF referring grasp synthesis on OCID-VLG.

### 12.1 Dataset and CLIP

```bash
cd "${VLMGP_ROOT}/crog_reproduction/CROG"
export PYC="${VLMGP_ROOT}/crog_reproduction/CROG/.venv-crog/bin/python"

"${PYC}" scripts/inspect_ocid_vlg.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml
```

Confirm that the configuration's data root points to the same OCID-VLG instance and that the CLIP RN50 hash matches Section 7.

### 12.2 Debug Training and Evaluation

```bash
"${PYC}" train_crog_mac.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml

"${PYC}" test_crog_mac.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml \
  --checkpoint exp/OCID-VLG_multiple_mac/CROG_mac_mps_debug/best_jindex_model.pth \
  --split val
```

The debug run must demonstrate that batches load, the loss is finite, gradients update, the checkpoint reloads, and grasp-rectangle coordinates and angle conventions agree.

### 12.3 Formal MPS Training

Local configuration:

```text
config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml
```

Key parameters are input 416, 50 epochs, batch 8, Validation batch 24, LR 1e-4, milestone 35, and seed 0. **The current configuration may retain an old `resume` path.** Copy it, set `resume: null`, and select a new output directory before starting. Never overwrite the old training run.

```bash
export CROG_ID="CROG_mps_repro_$(date -u +%Y%m%dT%H%M%SZ)"
export CROG_CONFIG="config/OCID-VLG/${CROG_ID}.yaml"
export CROG_OUTPUT="${VLMGP_ROOT}/crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/${CROG_ID}"
cp config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml "${CROG_CONFIG}"

# Manually point DATA.root_path in the YAML to the dataset; set TRAIN.exp_name
# to the actual CROG_ID value; verify TRAIN.output_folder; set TRAIN.resume: null.
test ! -e "${CROG_OUTPUT}"
rg -n 'root_path:|exp_name:|output_folder:|resume:' "${CROG_CONFIG}"
"${PYC}" train_crog_mac.py --config "${CROG_CONFIG}"
```

YAML does not expand the literal `${CROG_ID}`. `TRAIN.exp_name` must contain the actual ID emitted by the shell. `train_crog_mac.py` resolves its output directory as `TRAIN.output_folder/TRAIN.exp_name` and permits an existing directory. Copying the configuration without changing `exp_name` would continue writing into the old formal directory. The `test ! -e` check above is the pre-launch hard guard.

The local formal source checkpoint is:

```text
crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/
└── CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth
```

SHA-256:

```text
ac1304da520fbd3f6998dea88ba1e63b39b596b775856b39e5360a29576f1ddf
```

This checkpoint was trained locally under the `multiple` Train/Validation protocol and selected with Validation. The unified fair run subsequently froze it and exported candidates on the shared `unique` Test set of 7,675 samples. It is neither an official downloadable checkpoint nor a CROG model retrained on `unique` Train.

### 12.4 Official CUDA Route

The official repository README reports approximately 3.5 hours of training on 2×RTX 4090. This is an upstream reference, not a timing guarantee for the local MPS route. The official configuration filename in the local source is lower-case:

```bash
python -u train_crog.py \
  --config config/OCID-VLG/crog_multiple_r50.yaml
```

Reproducing the dissertation's formal `unique` fair panel requires the corresponding frozen split/candidate contract. Running the upstream `multiple` configuration directly reproduces only the upstream CROG baseline; it does not automatically reproduce the dissertation results.

### 12.5 Export Fair Candidates

After native CROG evaluation, the candidate export/correction tools are located under `crog_reproduction/CROG/failure_analysis/`; final fair generation is invoked by P1 of the unified entry point. Do not select a CROG threshold manually using Test labels. The unified run records candidate geometry, native score, and evaluator hash in its contract.

However, the dissertation replay in Section 13 explicitly binds the existing `fair_crog_hifics_g1_c1_no_rerank_...` source and `old-feature-dir`. It does not discover a new CROG checkpoint from Section 12 automatically. The dissertation's locked inputs reproduce the original re-ranking conclusion. Evaluating a new CROG model requires the failure-analysis/export tools to generate new Train/Validation/Test candidate sources, updated source-audit/old-feature inputs, and an entirely new fair experiment contract. Formal dissertation hashes cannot be reused.

## 13. Stage D: Unified Fair Candidate Re-ranking, P0–P15

This is the dissertation's core experiment. The sole authoritative operational guide is:

```text
runs/fair_unified_reranking_20260809_103012/README_REPRODUCE.md
```

### 13.1 Two Inviolable Rules

1. `run_all --resume` performs **only the P0–P1** audit, paired-manifest construction, and candidate-pool freezing. It does not execute the complete experiment.
2. At each step, execute only the first actionable `next_commands` entry reported by `_PIPELINE_STATUS.json`. Do not skip stages, and never execute Test again in an old locked run.

### 13.2 Initialise a New Run

```bash
cd "${VLMGP_ROOT}"
export PY4="${VLMGP_ROOT}/HiFi_reproduction/.venv-grasp4dof/bin/python"
export UNIFIED_ID="fair_unified_reranking_repro_$(date -u +%Y%m%dT%H%M%SZ)"
export UNIFIED_RUN="${VLMGP_ROOT}/runs/${UNIFIED_ID}"
export OLD_FEATURE_DIR="${VLMGP_ROOT}/runs/reranking_complete_20260803_094159/features"
export G1C1_BRIDGE_SOURCE="${VLMGP_ROOT}/HiFi_reproduction/runs/g1_c1_complete_reranking_20260806T084131Z"

test -d "${OLD_FEATURE_DIR}"
test -d "${G1C1_BRIDGE_SOURCE}"

"${PY4}" -m tools.unified_reranking.run_all \
  --run-dir "${UNIFIED_RUN}" \
  --fair-run "${VLMGP_ROOT}/runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523" \
  --modular-run "${MOD_SOURCE}" \
  --old-feature-dir "${OLD_FEATURE_DIR}" \
  --resume

"${PY4}" -m tools.unified_reranking.pipeline_status \
  --run-dir "${UNIFIED_RUN}" \
  --write-status

jq '{status,
     candidate_test_label_access_state,
     first_actionable_stage,
     next_commands: (.stages[.first_actionable_stage].next_commands // []),
     missing_or_invalid: (.stages[.first_actionable_stage].missing_or_invalid // [])}' \
  "${UNIFIED_RUN}/_PIPELINE_STATUS.json"
```

If `fair-run`, `modular-run`, or the old feature directory is absent or fails its hash check, P0 must fail closed. Do not continue by deleting the audit or changing an expected hash; restore the correct source assets.

`prepare_development.py`, `run_attribution_bridge.py`, and `prepare_test_bridge_bundle.py` also depend on the fixed G1/C1 bridge source above (fold assignments, historical candidates, and inventory), with some paths hard-coded in the source. A new HiFi/G1/C1 run does not replace it automatically. Changing the source requires a new bridge contract and a new lock.

### 13.3 P0–P1: Audit, Paired Manifests, and Candidate Freezing

Expected outputs include:

```text
00_audit/fair_source_audit.json
01_manifests/
02_candidates/candidate_contract_hashes.json
02_candidates/native_work/{crog,g1,c1}_{train,validation,test}/
commands.log
```

Verify that all three routes use the same sample identities; G1/C1 share the same HiFi mask; native scores are unmodified; candidate-geometry membership is frozen; and empty-output counts remain traceable.

### 13.4 P2–P5: Development Labels, Grouped Folds, and Features

Execute these stages in the order reported by status. They must satisfy all of the following:

- labels come only from Train/Validation;
- the same scene/frame cannot cross training and validation folds;
- candidate labels use the same evaluator as Section 1.2;
- features contain no GT-mask IoU, failure classification, or any quantity derived from Test labels;
- each candidate retains its route, sample, rank, geometry, native score, and source hash.

After each execution:

```bash
"${PY4}" -m tools.unified_reranking.pipeline_status \
  --run-dir "${UNIFIED_RUN}" --write-status
jq . "${UNIFIED_RUN}/_PIPELINE_STATUS.json"
```

Some P3/P4/P8/P9 items in `next_commands` are templates and may contain `<route>`, `<split>`, or `...`. Never paste these placeholders literally into a shell. Use the formal source ledger to inspect the concrete expansion for that stage/substage, then parameterise the repository root and new run paths. Do not execute the ledger in bulk:

```bash
export STAGE_TO_EXPAND="P3_P4"
jq -r --arg stage "${STAGE_TO_EXPAND}" \
  'select(.stage==$stage and .status=="COMPLETE") |
   [.substage,.command,.artifact_path] | @tsv' \
  "${VLMGP_ROOT}/runs/fair_unified_reranking_20260809_103012/commands.log"
```

For every expanded command, inspect `--help`, input hashes, and the output directory before executing it against the new `UNIFIED_RUN`. If the source ledger has no matching substage, stop and record a blocker rather than guessing arguments.

### 13.5 P6–P10: Calibration, Model Matrix, Validation Selection, Gating/Routing

Continue to run only `next_commands`. These stages perform:

- P6: fit/calibrate candidate scores using the development set only;
- P7: execute the preregistered model, feature, loss, and seed matrix;
- P8: choose the primary ranker using Validation metrics;
- P9: select the conservative gate and route router/union;
- P10: construct the attribution bridge without reselecting the best configuration on Test.

“Conservative gating” is not post hoc example selection. It is a Validation-frozen strategy: switch away from native Top-1 only when the selector is sufficiently confident; otherwise preserve the native choice.

### 13.6 P11: Pre-lock Assembly and Formal Lock

Execute these commands only when status reports that P11 is ready:

```bash
"${PY4}" -m tools.unified_reranking.prepare_test_bridge_bundle \
  --run-dir "${UNIFIED_RUN}"

"${PY4}" -m tools.unified_reranking.assemble_prelock \
  --run-dir "${UNIFIED_RUN}"

"${PY4}" -m tools.unified_reranking.create_formal_test_lock \
  --run-dir "${UNIFIED_RUN}" \
  --evaluation-plan "${UNIFIED_RUN}/08_lock/formal_evaluation_plan.json" \
  --extra-locked-file "code_manifest=${UNIFIED_RUN}/08_lock/code_manifest.json" \
  --extra-locked-file "prelock_assembly=${UNIFIED_RUN}/08_lock/prelock_assembly_manifest.json"
```

Then verify:

```bash
test -f "${UNIFIED_RUN}/08_lock/FORMAL_TEST_LOCK.json"
jq . "${UNIFIED_RUN}/08_lock/FORMAL_TEST_LOCK.json"
"${PY4}" -m tools.unified_reranking.pipeline_status \
  --run-dir "${UNIFIED_RUN}" --json \
  | jq -e '.first_actionable_stage=="P12" and .stages.P12.status=="READY"'
```

Do not use `--require-ready P12` to check a P12 stage that is READY but not yet complete. The current implementation treats missing P12 outputs as failure. The JSON assertion above verifies that P12 is exactly the next stage.

### 13.7 P12: Exactly-once Formal Test Execution

This is an irreversible scientific lifecycle point. Before starting, back up the lock file and hashes and confirm that no person or process has executed Test in this run:

```bash
test ! -f "${UNIFIED_RUN}/09_formal_test/FORMAL_TEST_EXECUTION.json"
```

Only after all preceding checks pass, execute:

```bash
"${PY4}" -m tools.unified_reranking.run_formal_test_once \
  --run-dir "${UNIFIED_RUN}"
```

If the command is interrupted, do not delete the execution marker and rerun it. Inspect the ledger/status recovery semantics first. The tool is designed for exactly-once execution; bypassing that protection invalidates the experiment.

### 13.8 P15 and post-formal Independent Recomputation

```bash
"${PY4}" -m tools.unified_reranking.independent_recompute \
  --run-dir "${UNIFIED_RUN}"

"${PY4}" -m tools.unified_reranking.build_postformal_artifacts \
  --run-dir "${UNIFIED_RUN}"

"${PY4}" -m tools.unified_reranking.pipeline_status \
  --run-dir "${UNIFIED_RUN}" --write-status
```

At completion, at least the following must hold:

```text
source_runs_immutable                         PASS
candidate_membership_geometry_invariance     PASS
native_score_preserved                       PASS
g1_c1_shared_hifi_masks                      PASS
no_gt_feature_leakage                        PASS
no_scene_frame_split_leakage                 PASS
no_candidate_test_access_before_lock         PASS
formal_test_only_once                        PASS
independent_recompute                        PASS
failure_taxonomy_denominator                 PASS
```

### 13.9 Trace Every Step with `commands.log`

The formal source run contains 1,814 JSONL entries. It is the complete command/artefact/hash ledger; this guide does not need to reproduce all 1,814 lines:

```bash
export UNIFIED_SOURCE="${VLMGP_ROOT}/runs/fair_unified_reranking_20260809_103012"

jq -r 'select(.status=="COMPLETE") |
  [.stage,.substage,.command,.artifact_path,.artifact_sha256] | @tsv' \
  "${UNIFIED_SOURCE}/commands.log" \
  > "${REPRO_ROOT}/formal_command_ledger.tsv"

awk -F '\t' '{count[$1]++} END {for (stage in count) print stage, count[stage]}' \
  "${REPRO_ROOT}/formal_command_ledger.tsv" | sort
```

The old commands contain absolute paths and may be used only as argument sources. A new run must be driven by the current `pipeline_status`; never run `bash commands.log`.

## 14. Stage E: Reproduce D1 (Dex-Net/GQ-CNN) Separately

D1 is neither a language model nor a 6-DoF pose generator. On the candidate side, the HiFi mask constrains the target region, Dex-Net generates geometric candidates, and GQ-CNN scores them. A separate retrospective run then re-ranks the frozen candidates. D1 must be labelled `D1_COMPATIBILITY` or `D1_RETROSPECTIVE`; it must not be merged into an unscoped table with FORMAL_PRIMARY.

### 14.1 Download the Model and Build the Container

```bash
cd "${VLMGP_ROOT}"
docker build --platform linux/amd64 \
  -t vlmgrasp/gqcnn-score:1.3.0 \
  -f HiFi_reproduction/docker/gqcnn-score/Dockerfile \
  HiFi_reproduction

mkdir -p "${VLMGP_ROOT}/HiFi_reproduction/models/gqcnn-official"
docker run --rm --platform linux/amd64 \
  -v "${VLMGP_ROOT}/HiFi_reproduction/models/gqcnn-official:/models:rw" \
  -v "${VLMGP_ROOT}/HiFi_reproduction/docker/gqcnn-score/download_gqcnn_2_1.sh:/tmp/download_gqcnn_2_1.sh:ro" \
  vlmgrasp/gqcnn-score:1.3.0 \
  bash /tmp/download_gqcnn_2_1.sh

export GQCNN_MODEL_ROOT="${VLMGP_ROOT}/HiFi_reproduction/models/gqcnn-official"
if test -f "${GQCNN_MODEL_ROOT}/model_zoo.zip"; then
  printf '%s  %s\n' \
    c3823f3525df851ea0b75c202e96e131ed85bafa9d4f3c4daa270a3c80943472 \
    "${GQCNN_MODEL_ROOT}/model_zoo.zip" | shasum -a 256 -c -
fi
printf '%s  %s\n' \
  eb5bc17089a39bd8fe6c801010c25a6a79a898d64181180feb5cf69aa630ff6f \
  "${GQCNN_MODEL_ROOT}/GQCNN-2.1/config.json" | shasum -a 256 -c -
test "$(find "${GQCNN_MODEL_ROOT}/GQCNN-2.1" -type f | wc -l | tr -d ' ')" = 22
```

The download script depends on `/opt/gqcnn` inside the container and must not be executed directly on a macOS host. The script itself does not verify hashes. The commands above verify the fallback archive (which may not exist when the old official script succeeds directly), `config.json`, and the file count. During D1 initialisation, a canonical JSON manifest of relative paths, file SHA values, and byte counts is calculated for the 22 files; its required SHA-256 is `8201961abe3a09d90c6c66e582a3bfeb181d7095a2ebcc3a9d90e68fc12e8614`. GQ-CNN's terms restrict use to educational, research, and not-for-profit purposes. Review the current upstream licence before use.

The exact source run also locked this Docker image ID:

```text
sha256:3d1158ca83197d55808454b718d0a328d3f27c57c80baaaea7031e21a9134ebd
```

Even an unchanged Dockerfile may produce a different image ID when rebuilt because the base image or package index has changed; D1 initialisation rejects this by design. Exact D1 reproduction requires restoration of the locked image archive. Accepting a new image requires a new, recorded D1 experiment contract and cannot be described as a bitwise replay.

### 14.2 Generate/Score the Compatibility Source (No Re-ranking)

The orchestrator is:

```text
HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py
```

It strictly verifies the dataset manifest, HiFi checkpoint, GQ-CNN model, Docker image, and input images. Initialise **only a new directory**:

```bash
export D1_RUN="${VLMGP_ROOT}/HiFi_reproduction/runs/d1_repro_$(date -u +%Y%m%dT%H%M%SZ)"

"${PY4}" HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py \
  --run-dir "${D1_RUN}" \
  --initialize \
  --initialize-only
```

After reviewing the generated manifest and plan, execute:

```bash
"${PY4}" HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py \
  --run-dir "${D1_RUN}" \
  --resume
```

Do not add `--initialize` again for an existing directory. The preregistered order is mask export → Dex-Net candidates → GQ-CNN scoring → corrected compatibility evaluation/report. The source protocol explicitly prohibits a re-ranker or training. It produces only the D1 candidate/score source and does not produce the dissertation's +18.619 pp re-ranking result.

### 14.3 Run the Independent D1 Retrospective Re-ranking

The authoritative source of the dissertation's D1 headline is:

```text
runs/fair_d1_reranking_extension_20260811T145515Z/
```

This run has 988 entries in `commands.log`. Its protocol is `fair-d1-reranking-retrospective-extension-v1`, explicitly identifying it as a retrospective extension rather than a pristine blind Test experiment. It binds:

```text
runs/fair_unified_reranking_20260809_103012
HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528
canonical evaluator SHA-256 f5155590b0b8d9f0748ad463688edfe6ca595d8ab7239ef5e29a5c595aeef301
```

First verify the static P0–P17 DAG and source:

```bash
cd "${VLMGP_ROOT}"
export D1_RETRO_SOURCE="${VLMGP_ROOT}/runs/fair_d1_reranking_extension_20260811T145515Z"

test -f "${D1_RETRO_SOURCE}/COMPLETE"
PYTHONPATH=src "${PY4}" -m tools.d1_reranking.render_command_dag \
  --check --format markdown
jq . "${D1_RETRO_SOURCE}/pipeline_status.json"
```

The bootstrap example currently generated by `render_command_dag` may still contain the unsupported `--evaluator` argument for the current `bootstrap` CLI. The DAG is authoritative only for stage ordering; its example commands must not be copied. Use the initialisation command below, which was checked against the current `--help`.

To create a new retrospective run:

```bash
export D1_RETRO_RUN="${VLMGP_ROOT}/runs/fair_d1_reranking_repro_$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONPATH=src "${PY4}" -m tools.d1_reranking.bootstrap \
  --run-dir "${D1_RETRO_RUN}" \
  --unified-run "${VLMGP_ROOT}/runs/fair_unified_reranking_20260809_103012" \
  --snapshot-a "${VLMGP_ROOT}/HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
```

Thereafter, treat the new `pipeline_status.json` field `first_incomplete_stage` and the static DAG as the ordering authority, and expand the commands for the same stage from the source ledger. D1 has neither a `pipeline_status` CLI nor a verified one-command end-to-end driver. The recovery path below is stage-by-stage and requires manual verification of `--help` and input hashes:

```bash
export D1_STAGE="P1"
jq -r --arg stage "${D1_STAGE}" \
  'select(.stage==$stage and .status=="COMPLETE") |
   [.substage,.command,.artifact_path,.artifact_sha256] | @tsv' \
  "${D1_RETRO_SOURCE}/commands.log"
```

Do not execute source absolute paths line by line and do not write into the completed source run. After P0–P13 are complete and all pre-lock conditions are met, the formal tail uses this tool order:

```bash
PYTHONPATH=src "${PY4}" -m tools.d1_reranking.assemble_formal_evaluation_plan \
  --run-dir "${D1_RETRO_RUN}" --resume

PYTHONPATH=src "${PY4}" -m tools.d1_reranking.create_formal_lock \
  --run-dir "${D1_RETRO_RUN}" \
  --evaluation-plan "${D1_RETRO_RUN}/configs/d1_formal_evaluation_plan.json"

# Exactly once. Verify pipeline_status/lock before execution; never rerun the source run.
PYTHONPATH=src "${PY4}" -m tools.d1_reranking.run_formal_test_once \
  --run-dir "${D1_RETRO_RUN}"

PYTHONPATH=src "${PY4}" -m tools.d1_reranking.run_independent_with_source_adapter \
  --run-dir "${D1_RETRO_RUN}"

PYTHONPATH=src "${PY4}" -m tools.d1_reranking.build_postformal_with_source_adapter \
  --run-dir "${D1_RETRO_RUN}"

PYTHONPATH=src "${PY4}" -m tools.d1_reranking.finalize_with_source_adapter \
  --run-dir "${D1_RETRO_RUN}"

test -f "${D1_RETRO_RUN}/COMPLETE"
test -f "${D1_RETRO_RUN}/FINAL_RUN_LOCK.json"
test -f "${D1_RETRO_RUN}/17_independent_recompute/recomputed_metrics.json"
```

P1–P13 include source reconciliation, candidate freezing, development labels, folds, features, calibration, the main model matrix, gate, K-sensitivity, ablations, and four-route extension. They must be driven by the new run's resource authorisation and lock state. The source ledger also records failed retries and reconciliations; all 988 entries cannot simply be executed.

### 14.4 Two Non-interchangeable D1 Contracts

- `D1_COMPATIBILITY`: source-native GQ-CNN all-NMS order and compatibility evaluator;
- `D1_RETROSPECTIVE`: frozen post-NMS Top-5 candidates and retrospective re-ranking with the same evaluator SHA as the formal primary panel.

The dissertation's +18.619 percentage-point D1 result belongs to `D1_RETROSPECTIVE`. Compatibility figures of approximately 0.4627/0.7053/0.8377 in the repository belong to another contract and cannot replace it.

## 15. Stage F: Rebuild Evidence, Figures, and the Dissertation

### 15.1 Rebuild Derived Evidence Only

Source evidence bundle:

```text
runs/four_route_evidence_consolidation_20260813T155455Z/
```

Its `10_reproducibility/README_REPRODUCE.md` permits rebuilding derived assets only from frozen outputs. It prohibits training, candidate generation, threshold selection, formal Test execution, or writing a D1 finaliser back into a source run.

```bash
cd "${VLMGP_ROOT}"
export EVIDENCE_RUN="${VLMGP_ROOT}/runs/four_route_evidence_consolidation_20260813T155455Z"
export PYEVIDENCE="${VLMGP_ROOT}/HiFi_reproduction/.venv-grasp4dof/bin/python"

"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/build_bundle.py"
```

The presentation build requires a Node runtime. The bundled runtime verified in the current Codex workspace is shown below. On another machine, replace it with a compatible Node installation and set `NODE_PATH` to a directory containing `pptxgenjs` and the other dependencies:

```bash
export CODEX_NODE="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
export NODE_PATH="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"

"${CODEX_NODE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/build_deck.mjs"

soffice --headless --convert-to pdf \
  --outdir "${EVIDENCE_RUN}/09_presentation" \
  "${EVIDENCE_RUN}/09_presentation/Four_Route_Evidence_Consolidation.pptx"
```

First compile the dissertation-fragment check bundled with the evidence; `verify_bundle.py` explicitly requires its PDF. This differs from the complete dissertation compilation in the next subsection:

```bash
cd "${EVIDENCE_RUN}/08_thesis"
latexmk -lualatex -interaction=nonstopmode -halt-on-error \
  thesis_compile_check.tex
cd "${VLMGP_ROOT}"
```

Then run the snapshot and integrity checks:

```bash
"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/source_snapshot.py" after
"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/source_snapshot.py" compare
"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/verify_bundle.py"
```

System `python3` lacks the build dependencies. `.venv-paper-tools` contains `pandas`/`matplotlib` but currently lacks `pyarrow`, so it cannot read Parquet. The `.venv-grasp4dof` environment used above has been verified to contain `pandas`, `matplotlib`, and `pyarrow`, and is therefore reused for the evidence scripts.

### 15.2 Compile the Dissertation

The dissertation template uses LuaLaTeX/Biber. Do not substitute pdfLaTeX/BibTeX:

```bash
cd "${VLMGP_ROOT}/MSc_Dissertation_Template_the_University_of_Manchester_EEE__2025_onwards"

latexmk -g -lualatex -shell-escape \
  -interaction=nonstopmode \
  -halt-on-error \
  main.tex
```

Success requires exit code 0, no remaining undefined citations, agreement between machine-readable tables and the prose values, and a generated `main.pdf`. To check only the evidence fragment, enter the evidence bundle's `08_thesis/` directory and build `thesis_compile_check.tex` according to its README.

## 16. How to Determine Whether Reproduction Succeeded

Do not validate reproduction from one mean alone. Apply the following hierarchy:

1. **Identity**: record Git commits, third-party commits, configurations, dataset manifests, sample order, checkpoints, and evaluator SHA values;
2. **Data**: `unique` Train/Validation/Test contain 26,295/3,778/7,675 samples, with no scene/frame leakage into development;
3. **Lifecycle**: Test labels were not accessed before the lock, and formal Test execution count is exactly 1;
4. **Candidate contract**: CROG/G1/C1 each use route-local Top-5 candidates, with native scores and geometry unmodified;
5. **Metric**: one GT must satisfy both IoU and angle criteria; empty outputs remain in the denominator;
6. **Directionality**: gated ΔJ@1 is positive for all three formal routes, and recovered cases substantially outnumber harmful cases;
7. **Statistics**: the scene-cluster bootstrap 95% lower bound is > 0, and the Holm-corrected McNemar p-value is < 0.05;
8. **Independent recomputation**: all values agree with the formal artefacts;
9. **Reporting**: tables, figures, summaries, and the dissertation PDF are generated from machine-readable artefacts, without manual transcription drift;
10. **Claim boundary**: claim only improvement in the offline 4-DoF proxy, not physical success or complete 6-DoF execution.

Permitted result-reporting format:

```text
Status: COMPUTATIONALLY REPRODUCED / PARTIALLY REPRODUCED / NOT REPRODUCED
Code commit: ...
Dataset manifest-set SHA-256: ...
Device and software: ...
Formal Test execution count: 1
Routes: CROG/G1/C1
N: 7,675
Native J@1: ...
Gated J@1: ...
ΔJ@1: ...
Recovered/Harmful: ...
Independent recomputation: PASS/FAIL
Differences from the reference and reasons: ...
```

## 17. What Is Currently Missing from a Fresh Public Clone

At the audited commit, `git ls-files` returns zero entries for the following core paths. `.gitignore` also excludes data, models, checkpoints, `*.pth/*.pt`, and all `**/runs/` directories:

```text
runs/fair_unified_reranking_20260809_103012/
runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523/
runs/fair_d1_reranking_extension_20260811T145515Z/ (D1 retrospective 988-entry ledger, locks, and results)
runs/reranking_case_visuals_v2_20260810T203154Z/ (evidence-bundle case-board source)
runs/reranking_complete_20260803_094159/features/ (P1 frozen CROG Train/Validation candidate source)
runs/four_route_evidence_consolidation_20260813T155455Z/
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/
HiFi_reproduction/runs/hifics_ocidvlg_20260711_112921/ (D1 also depends on the old single-FiLM template/predictions)
HiFi_reproduction/runs/grasp_backend_comparison_20260807_090155/ (evidence-bundle COMP_DIAG source)
HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/
HiFi_reproduction/runs/modular_reranking_repeatedfilm_v1_20260729_203147/
HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_20260803_092055/ (G1/C1 training-grid, manifest, and parent-ledger source)
HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/
HiFi_reproduction/runs/g1_c1_complete_reranking_20260806T084131Z/ (fold and attribution/Test bridge source)
HiFi_reproduction/hifics/
HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml
HiFi_reproduction/artifacts/data_audit/frozen_manifests/
HiFi_reproduction/artifacts/experiment/hierarchical_film_tests.json (required by the old-baseline comparison report; rebuildable from source)
HiFi_reproduction/reports/hifics_previous_checkpoint_recomputed_protocols.json (required by the old-baseline comparison report)
HiFi_reproduction/third_party_src/grconvnet/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98 (restorable from the official pinned URL in Section 7.2)
HiFi_reproduction/third_party_src/checkpoints/ggcnn2/ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt
HiFi_reproduction/models/gqcnn-official/
crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth
OCID-VLG data
```

To enable another researcher to complete R2/R3 from the public repository, publish a versioned reproduction asset package with an overall SHA-256. At minimum, it should include:

1. dataset download instructions, the locally recorded original-ZIP hash, and the three frozen manifests (the dataset itself need not be redistributed);
2. the pinned HiFi upstream commit and download script, project-owned patch/diff, controlled configuration, environment lock, best checkpoint, and snapshot-hash manifest. Retain the complete upstream source snapshot only as a private/local audit artefact unless the authors explicitly grant redistribution permission;
3. official GR-ConvNet/GG-CNN2 weight sources, complete hashes, and the G1/C1 fine-tuned checkpoints;
4. the exact CROG configuration, environment lock, formal checkpoint, and candidate-export manifest;
5. the fair native source run and unified run, including locks, ledgers, required Parquet files, and reports;
6. the D1 GQ-CNN/Dex-Net input contract, model hashes, and a verifiable archive of the locked Docker image, if exact D1 reproducibility is claimed;
7. the evidence bundle and dissertation-referenced figures;
8. licence/redistribution permission for every third-party asset. For components without a licence or without redistribution permission, publish only the download script, pinned commit, and patch, not the source or weights.

The formal unified source run recorded repository baseline `522b5f2156e7ee7bb02ac3aba535aa58ec890ca7` plus `00_audit/current_git_diff.patch`, whereas the audited public HEAD for this guide is `461505999123fe23303bab45bfc7ddbf9619f1fb`. Exact result provenance must follow the formal lock/source audit; the current clean HEAD alone is not a substitute.

## 18. Common Failures, Causes, and Recovery Order

### 18.1 OCID-VLG Not Found

```bash
ls -ld "${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG"
ls "${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG/refer/unique"
```

A common cause is an extra extraction directory or a symbolic link pointing to a missing location. Find the directory that actually contains `ARID10/ARID20/refer`, then rebuild the links. Do not make three copies.

### 18.2 Manifest Count or Hash Mismatch

Possible causes include a different dataset version, relative-path normalisation, sample ordering, or local manifest-generation code. Stop formal training and compare each record and ordered sample ID. Do not merely change the expected hash to the new value.

### 18.3 MPS Unavailable or Operator Failure

Run the small-sample checks and related unit tests first. If falling back to CPU, record the device change. If moving to CUDA, create a new environment and report numerical differences. Never change devices part-way through training without recording it.

### 18.4 Incomplete Results after `run_all --resume`

This is expected: it stops after P1. Run `pipeline_status --write-status`, then execute the first `next_commands` entry from `_PIPELINE_STATUS.json`.

### 18.5 Formal Test Rejects a Second Execution

This is a safety mechanism, not a bug. Create a fresh run and execute it from the beginning. Do not delete the lock or execution marker.

### 18.6 G1/C1 Results Differ Greatly from Older Reports

Inspect the candidate pool, selector, angle sign, NMS, and evaluator contract first. Unified fair results and historical modular results are not directly comparable.

### 18.7 Docker/GQ-CNN Failure

Verify the Docker daemon, `linux/amd64` emulation, inner and outer model hashes, and read-only mounts. Native macOS with a modern Python version is not an equivalent replacement for the TensorFlow 1.15 environment.

### 18.8 LaTeX Compilation Failure

Confirm that LuaLaTeX is used, Biber and the required fonts/packages are installed, and `-shell-escape` is enabled. Compile the minimal `thesis_compile_check.tex` first to isolate the problem. Do not edit generated table values manually to bypass a build failure.

## 19. Optional Extensions: Do Not Confuse Them with the Main Reproduction

### 19.1 Florence-2 + SAM Demo

This applies only to `target_aware_vlm_grasping`. The default expected locations are:

```text
target_aware_vlm_grasping/models/vlm/florence2-large-ft/
target_aware_vlm_grasping/models/vlm/sam/sam_vit_b_01ec64.pth
```

Pin a Hugging Face revision for Florence-2 and prefer safetensors. Use Meta's official checkpoint for SAM ViT-B. When `trust_remote_code=True` is involved, pin the revision and review the code before execution.

### 19.2 GraspNet/6-DoF

The official entry point is the [GraspNet dataset page](https://graspnet.net/datasets.html). This repository provides:

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python -m graspnet6d.cli all \
  --profile paper-lite \
  --run-id "graspnet6d_$(date -u +%Y%m%dT%H%M%SZ)"
```

After interruption, use the documented `--resume` only for the same run. A new experiment requires a new run ID. The project uses a `src/` layout and is not installed as a site package; omitting `PYTHONPATH=src` in a clean shell produces `ModuleNotFoundError`.

Read `docs/MANUAL_DOWNLOAD_REQUIRED.md` and `src/graspnet6d/compact_download.py` first. The current local 30/3-scene subset can support development checks only. It is not the official GraspNet benchmark and cannot support a claim of physical grasp success.

### 19.3 VGN/LAVT/VL-Grasp/AnyGrasp

These components can support later comparisons or three-dimensional extensions, but their licences, data contracts, coordinate systems, and metrics differ. Every new experiment requires a separate scope, manifest, evaluator, and results table. It must not be appended to the FORMAL_PRIMARY rows.

## 20. Claim–Evidence–Source Map

| Claim | Machine-readable/dissertation evidence | External source |
| --- | --- | --- |
| OCID-VLG is a language-conditioned 4-DoF grasping dataset | Frozen `unique` manifests; dissertation Chapters 3/4 | [Official OCID-VLG repository](https://github.com/gtziafas/OCID-VLG) |
| CROG is end-to-end CLIP referring grasp synthesis | CROG configuration/checkpoint/candidate manifest | [CROG PMLR](https://proceedings.mlr.press/v229/tziafas23a.html), [code](https://github.com/HilbertXu/CROG) |
| HiFi-CS supplies only the repeated-FiLM target mask in this dissertation | HiFi source snapshot, configuration, run summary | [HiFi-CS repository](https://github.com/vineet2104/hifics), [paper](https://arxiv.org/abs/2409.10419) |
| G1/C1 are modular back ends sharing a HiFi mask | G1/C1 selected configurations and source audit | [GR-ConvNet](https://github.com/skumra/robotic-grasping), [GG-CNN](https://github.com/dougsm/ggcnn) |
| Formal Test is `unique`, N=7,675, and executed once | FORMAL_TEST_LOCK, FORMAL_TEST_EXECUTION, ledger | Official OCID-VLG split description |
| Re-ranking improves J@1 on all three formal routes | Final results, scene bootstrap, McNemar, independent recomputation | Formal repository run evidence |
| D1 must be reported separately | Scope-contract map, D1 manifests | [GQ-CNN](https://github.com/BerkeleyAutomation/gqcnn) |
| The offline rectangle metric is not physical success | Evaluator contract and absence of ROS/hardware-execution evidence | The CROG paper's robot experiments are a different experimental system |

## 21. External Resources and Reuse Decision

This reproduction prioritises formal repository code, locks, and ledgers. External implementations are used only as pinned upstream model/data sources:

- OCID-VLG: use the authors' dataset and official split; do not copy a third-party mirror;
- CROG: pin the MIT source commit, then train/export with the repository's local adaptation;
- HiFi-CS: pin and refer to the authors' implementation, but do not repackage it because no repository-level licence was found. Manage dissertation modifications through a project-owned patch and private/local source-snapshot audit;
- GR-ConvNet and GG-CNN/VGN: retain their upstream BSD-3-Clause source licences and commits; confirm weight redistribution rights separately;
- OpenAI CLIP: retain the pinned MIT source-code commit; do not infer a separate pretrained-weight licence from the source-code licence;
- GQ-CNN: use it locally only under its educational, research, and not-for-profit terms;
- no external implementation was found that can legally, safely, and losslessly replace this repository's P0–P15 locked re-ranking system. Core re-ranking therefore uses repository-owned code, with no code copied from blogs or Stack Overflow.

## 22. Final Checklist

### Acquisition and Identity

- [ ] Pin the VLMGraspPose commit and preserve the dirty diff
- [ ] Read `docs/THIRD_PARTY_NOTICES.md`
- [ ] Pin every upstream commit
- [ ] Download OCID-VLG and preserve the locally recorded ZIP SHA-256
- [ ] Verify the `unique` manifest counts and hashes
- [ ] Download each weight and verify its complete SHA-256

### Training and Candidates

- [ ] Pass the HiFi overfit smoke check
- [ ] Complete HiFi full training, Validation selection, one route-local HiFi Test, and independent verification
- [ ] Pass the G1/C1 source and dataset audits
- [ ] Select and lock G1/C1 configurations using Validation only
- [ ] Complete CROG debug/full training, checkpointing, and candidate export
- [ ] Pass the three-route paired-sample-identity and Top-5 candidate contracts

### Re-ranking and Formal Testing

- [ ] Drive P0–P10 in order through `pipeline_status`
- [ ] Confirm no GT/Test feature leakage or scene/frame leakage
- [ ] Complete the P11 pre-lock and formal lock
- [ ] Confirm that P12 execution count is exactly 1
- [ ] Complete P15 independent recomputation and ensure that all integrity checks report PASS
- [ ] If D1 is run, report compatibility/retrospective results separately

### Writing and Reporting

- [ ] Report FORMAL_PRIMARY N, native, gated, gain, recovered/harmful, and statistical intervals
- [ ] Do not mix old `multiple`, legacy-evaluator, or D1 results into the primary table
- [ ] Do not describe offline 4-DoF J@1 as 6-DoF or physical success
- [ ] Pass evidence-bundle verification
- [ ] Compile the LuaLaTeX dissertation successfully
- [ ] Record every difference from the reference and every asset that remains unreproduced

---

If the objective is only to confirm that the code works, complete Sections 5–9. To regenerate the dissertation's primary numerical results, complete Sections 6–13 and 15–18. To enable a third party to reproduce the complete work independently from the public GitHub repository, first publish the assets in Section 17.
