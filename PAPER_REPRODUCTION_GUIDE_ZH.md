# VLMGraspPose 论文完整复现指南（中文）

> 论文/学位论文：**From Prediction to Selection: Re-Ranking Grasp Candidates in End-to-End and Modular Pipelines for Language-Guided Robotic Grasping**  
> 仓库：[langchengg/VLMGraspPose](https://github.com/langchengg/VLMGraspPose)  
> 本指南审计基线：`461505999123fe23303bab45bfc7ddbf9619f1fb`  
> 审计日期：2026-08-27  
> 当前复现状态：**PARTIALLY VERIFIED（部分验证）**

## 0. 先读结论：这里的“完整复现”到底是什么

本仓库最终论文的正式贡献不是从 RGB-D 直接输出机器人可执行 6-DoF 位姿，也不是复现 CROG 原论文中的实体机器人成功率。它完成的是：

1. 在 OCID-VLG 官方 `unique` 划分上训练语言指代分割模型；
2. 用三条 4-DoF 平面抓取路线生成候选：端到端 CROG、模块化 HiFi-CS→GR-ConvNet（G1）、HiFi-CS→GG-CNN2（C1）；
3. 将每条路线冻结后的 Top-5 候选置于同一评估合同下；
4. 只用 Train/Validation 标签训练、选择和门控候选重排序器；
5. 锁定代码、模型、候选池、阈值和评估计划后，对 7,675 个 Test 样本只执行一次正式 Test 评估；
6. 独立重算结果并生成论文表格、图、报告与学位论文 PDF。

正式主面板是 **CROG/G1/C1 的离线 4-DoF 候选重排序**。D1（HiFi-CS→Dex-Net/GQ-CNN）是单独的回顾性/兼容性面板；GraspNet、VGN、AnyGrasp、VL-Grasp、LAVT、SAM/Florence 等属于次级原型、诊断或未来 6-DoF 扩展，不能与正式主结果混报。

### 0.1 可复现等级

| 等级 | 能做什么 | 当前工作区 | 从公开仓库全新克隆 |
| --- | --- | --- | --- |
| R0 源码审计 | 阅读代码、测试、论文和合同 | 可做 | 可做 |
| R1 证据重建 | 从已冻结正式输出重建表格、图和论文 | 可做 | **不可做**：正式 `runs/` 未提交 |
| R2 计算复现 | 重新训练模型、生成候选、重排序并正式测试 | **部分可做**：主面板冻结来源可重放统一重排序，HiFi/CROG 可独立重训；G1/C1 初始化账本及精确 D1 依赖不全 | **不可完整做**：数据、权重、冻结清单及若干源码快照未发布 |
| R3 独立/近比特复现 | 按哈希获得同一输入并重算结果 | 本机保留资产可验证 | 当前不可做，需先发布第 17 节清单中的资产 |

因此，本指南同时提供两条路线：

- **路线 A：当前本机工作区**。可核验已完成实验，并可用现存冻结 G1/C1 来源资产在新 run 中重跑统一重排序；HiFi/CROG 可独立重训。G1/C1 从空目录开始的完整初始化账本缺失，旧 single-FiLM/精确 D1 Docker 镜像资产也不完整；这些路线只能按各自现有合同建立新实验，不能称为原运行的逐命令重放。
- **路线 B：全新公开克隆**。可完成环境、数据、上游源码、单元测试和次级 demo；在缺失资产补齐前，不能声称复现出论文正式数值。

## 1. 论文问题、输入、输出和成功标准

### 1.1 输入与输出

单个 OCID-VLG 样本包含 RGB 图像、深度图、自然语言指代表达、目标框/掩码和一个或多个真实抓取矩形。官方数据说明见 [OCID-VLG](https://github.com/gtziafas/OCID-VLG)。

本论文的抓取候选为平面矩形：

```text
g = (center_x, center_y, width, height, angle)
```

它是相机图像平面中的 4-DoF/矩形抓取代理，不包含完整的三维平移、三维旋转、机械臂逆运动学、碰撞轨迹或真实闭环执行。

### 1.2 唯一正确的正式评估合同

预测成功当且仅当存在**同一个**真实抓取矩形，同时满足：

```text
rotated IoU > 0.25
AND
180° 周期角误差 <= 30°
```

注意：

- IoU 是严格大于 `0.25`，不是大于等于；
- 角度按夹爪的 180° 对称性计算；
- 两个条件必须由同一个 GT 矩形满足，不能用 GT-A 满足 IoU、GT-B 满足角度；
- 空候选/空预测仍留在分母中；
- Test 标签在正式锁完成前不得用于特征选择、阈值选择、模型选择或门控选择。

### 1.3 论文应复现的正式结果

下表来自当前锁定证据，不是重新运行后预先保证的精确浮点结果。硬件、库版本或非确定性内核可能引入差异；合同、样本身份和方向性结论必须一致。

| 路线 | 范围 | N | Native J@1 | 保守门控重排序后 J@1 | 增量（百分点） | 恢复/伤害 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CROG | FORMAL_PRIMARY | 7,675 | 0.892248 | 0.923648 | +3.140 | 278 / 37 |
| G1 | FORMAL_PRIMARY | 7,675 | 0.475179 | 0.566384 | +9.121 | 729 / 29 |
| C1 | FORMAL_PRIMARY | 7,675 | 0.438176 | 0.562476 | +12.430 | 1,006 / 52 |
| D1 | D1_RETROSPECTIVE，单列 | 7,675 | 0.328990 | 0.515179 | +18.619 | 1,596 / 167 |

“恢复”表示 native Top-1 失败而门控选择成功；“伤害”表示 native Top-1 成功而门控选择失败。正式主结论是：候选池已有可利用的 Top-K 余量时，保守选择器能提升 Top-1；它不证明离线矩形指标等同于物理抓取成功率。

### 1.4 不可混用的历史数字

仓库中还有旧实验，例如 CROG `multiple` Test 17,749 样本的历史结果，以及早期 G1/C1 角度符号/候选合同不同的结果。它们属于 `LEGACY_INCOMPATIBLE`，不能与 `FORMAL_PRIMARY` 合并。最终口径以以下文件为准：

- `runs/four_route_evidence_consolidation_20260813T155455Z/03_tables/table_01_scope_contract_map.md`
- `runs/four_route_evidence_consolidation_20260813T155455Z/03_tables/table_04_native_vs_gated.md`
- `runs/fair_unified_reranking_20260809_103012/08_lock/FORMAL_TEST_LOCK.json`
- `runs/fair_unified_reranking_20260809_103012/15_independent_recompute/INDEPENDENT_RECOMPUTE.json`

## 2. 从数据到论文的总流程

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

## 3. 仓库地图：每个目录在复现中做什么

| 路径 | 作用 | 是否属于正式主链 |
| --- | --- | --- |
| `MSc_Dissertation_Template_the_University_of_Manchester_EEE__2025_onwards/` | 学位论文 LaTeX 源码 | 是，最终写作 |
| `src/unified_reranking/`、`tools/unified_reranking/` | 公平候选生成、特征、模型、门控、锁、正式评估、独立重算 | **是，核心** |
| `tests/unified_reranking/` | 正式合同与生命周期测试 | 是 |
| `crog_reproduction/CROG/` | CROG 源码快照、Mac/MPS 适配、训练、评估和候选导出 | 是，CROG 路线 |
| `HiFi_reproduction/hifics/` | HiFi-CS 上游固定提交及本地 repeated-FiLM 修改 | 是，但整个目录被 Git 忽略 |
| `HiFi_reproduction/tools/grasp4dof/` | 数据合同、G1/C1 迁移/微调、候选、锁和报告 | 是，模块化路线 |
| `HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py` | D1 初始化、mask 导出、Dex-Net/GQ-CNN 编排 | 仅 D1 单列 |
| `src/d1_reranking/`、`tools/d1_reranking/` | D1 回顾性候选重排序 | 仅 D1 单列 |
| `src/gtmask_counterfactual/` | GT-mask 反事实诊断 | 否，post-formal 机制分析 |
| `target_aware_vlm_grasping/` | Florence-2/SAM + CPU 几何候选演示 | 否，次级原型/快速上手 |
| `src/graspnet6d/`、`legacy/external_graspnet/`、VGN/AnyGrasp 相关目录 | 6-DoF 子集和扩展实验 | 否，不能替代正式 4-DoF 主结果 |
| `LAVT_reproduction/` | GPL-3.0 指代分割基线 | 否，可选 |
| `runs/`、`HiFi_reproduction/runs/` | 锁定运行证据、候选、模型、图表和报告 | 本机是，公开仓库不包含 |

## 4. 能力与环境审计

### 4.1 当前机器已验证能力

当前工作区审计到：Apple M5 Pro、24 GiB 统一内存、PyTorch MPS 可用、无 CUDA；Docker CLI 存在但 daemon 未启动；没有 ROS、MoveIt 或 Gazebo。因此：

- 可用 MPS/CPU 重跑 HiFi-CS、G1、C1、CROG 适配版和统一重排序；
- GQ-CNN 的 TensorFlow 1.15/Python 3.7 路线应走 `linux/amd64` Docker，不能依赖当前原生 Python；
- 当前环境只能验证离线指标，不能执行机械臂抓取；
- 磁盘已较满，先运行 `df -h` 和 `du -sh`，不要重复下载现有数据/权重。

已存在并验证的隔离环境：

```text
HiFi_reproduction/.venv-grasp4dof   Python 3.11，Torch/MPS，主 4-DoF 与统一重排序
HiFi_reproduction/.venv-gqcnn       候选侧依赖；不是原生 TF1.15 运行环境
.venv-graspnet6d                     可选 6-DoF 子集工具
.venv-paper-tools                    绘图/论文辅助；当前缺少 pyarrow，不能直接读取 evidence Parquet
```

不要将所有模块强行安装到一个环境；CROG、HiFi-CS、4-DoF、GQ-CNN 和论文构建的依赖年代不同。

### 4.2 使用与未使用的能力

- 使用：仓库内测试、`commands.log` 账本、锁定 JSON/Parquet 证据、SHA-256、Git、现有虚拟环境、官方 GitHub/论文/模型页。
- 可选使用：Docker（仅 D1/GQ-CNN）、`gdown`（Google Drive 下载）、LuaLaTeX/`latexmk`（论文构建）。
- 不需要：ROS/MoveIt/Gazebo、在线/实体机器人执行、CUDA，除非转向官方 CROG CUDA 路线或物理扩展。
- 未用于正式复现：Florence/SAM demo、GraspNet/VGN/AnyGrasp、LAVT、通用绘图或数据分析插件；它们不控制论文主数字。

## 5. 建立干净、可审计的工作目录

### 5.1 全新克隆

```bash
git clone https://github.com/langchengg/VLMGraspPose.git
cd VLMGraspPose
git checkout 461505999123fe23303bab45bfc7ddbf9619f1fb

export VLMGP_ROOT="$(pwd)"
git status --short
git rev-parse HEAD
```

如果在当前本机仓库工作，只运行：

```bash
cd /path/to/VLMGraspPose
export VLMGP_ROOT="$(pwd)"
```

当前工作树已有用户自己的删除状态；不要使用 `git reset --hard`、`git clean -fdx` 或任何会抹掉数据、运行证据和用户修改的命令。

### 5.2 创建新的运行 ID

任何重跑都必须写入新目录，绝不覆盖 `runs/fair_unified_reranking_20260809_103012` 或其他带锁的来源运行：

```bash
export REPRO_ID="repro_$(date -u +%Y%m%dT%H%M%SZ)"
export REPRO_ROOT="${VLMGP_ROOT}/runs/${REPRO_ID}"
mkdir -p "${REPRO_ROOT}"
printf '%s\n' "${REPRO_ROOT}"
```

记录主机和 Git 状态：

```bash
uname -a > "${REPRO_ROOT}/host.txt"
git rev-parse HEAD > "${REPRO_ROOT}/git_head.txt"
git status --porcelain=v1 > "${REPRO_ROOT}/git_status.txt"
python3 --version > "${REPRO_ROOT}/python.txt"
```

## 6. 下载、校验和放置 OCID-VLG 数据

### 6.1 官方来源与许可边界

官方数据页是 [gtziafas/OCID-VLG](https://github.com/gtziafas/OCID-VLG)，下载链接指向作者提供的 [Google Drive 文件](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link)。官方说明数据含 89,639 个 image-text-mask-grasp 元组、1,763 个场景，并提供 `multiple`、`unique`、`novel-instances`、`novel-classes` 四种划分。

OCID-VLG 仓库未提供清晰的数据集级再分发许可证，也未公布该 Drive 压缩包的官方 SHA-256。默认只做本地学术使用并引用作者，不要把数据压缩包提交到本仓库或重新分发。

### 6.2 下载

优先在浏览器打开上述 Drive 链接。如果要用命令行，可在独立下载环境安装 `gdown`：

```bash
python3 -m venv "${VLMGP_ROOT}/.venv-download"
"${VLMGP_ROOT}/.venv-download/bin/python" -m pip install --upgrade pip gdown
mkdir -p "${VLMGP_ROOT}/downloads/ocid_vlg"

"${VLMGP_ROOT}/.venv-download/bin/gdown" \
  --fuzzy "https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link" \
  -O "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip"
```

若 Google Drive 报配额或确认页错误，停止重试，改用浏览器下载；不要使用来源不明的镜像。

### 6.3 下载后完整性检查

由于官方没有给压缩包哈希，第一次可信下载后应记录自己的哈希：

```bash
file "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip"
unzip -t "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip"
shasum -a 256 "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip" \
  | tee "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip.sha256"
```

不要把下列“冻结 manifest 哈希”误当成官方 ZIP 哈希；它们只验证本论文生成的样本清单：

| unique 清单 | 样本数 | SHA-256 |
| --- | ---: | --- |
| Train | 26,295 | `a986bcce3e1961be816a295c3ae0942e64e61275524a85c0a8957563e7f920c1` |
| Validation | 3,778 | `573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a` |
| Test | 7,675 | `915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409` |
| 三清单集合 | 37,748 | `fa36db4ff548f2ca06abadefb30cacc741165478dff8691203375a311b86c0c8` |

### 6.4 解压和目录结构

```bash
mkdir -p "${VLMGP_ROOT}/datasets/OCID-VLG"
unzip "${VLMGP_ROOT}/downloads/ocid_vlg/ocid_vlg.zip" \
  -d "${VLMGP_ROOT}/datasets/OCID-VLG"

find "${VLMGP_ROOT}/datasets/OCID-VLG" -maxdepth 2 -type d | sort | sed -n '1,80p'
```

如果压缩包额外包了一层目录，把 `OCID-VLG` 实际根目录记为 `OCIDVLG_ROOT`。根目录至少应能看到：

```text
OCID-VLG/
├── ARID10/
├── ARID20/
├── refer/
│   ├── multiple/
│   ├── unique/
│   ├── novel-instances/
│   └── novel-classes/
└── catalog.csv（若下载版本包含）
```

设置唯一来源并用符号链接供不同子项目消费：

```bash
export OCIDVLG_ROOT="${VLMGP_ROOT}/datasets/OCID-VLG"

ln -s "${OCIDVLG_ROOT}" "${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG"
ln -s "${OCIDVLG_ROOT}" "${VLMGP_ROOT}/crog_reproduction/OCID-VLG"
ln -s "${OCIDVLG_ROOT}" "${VLMGP_ROOT}/target_aware_vlm_grasping/data/OCID-VLG"
```

若目标路径已经存在，先用 `readlink`/`ls -ld` 核对，不要用 `ln -sf` 覆盖。CROG 配置中的数据根路径也应指向同一个 `OCIDVLG_ROOT`，以避免三份数据漂移。

### 6.5 数据清单核验

当前本机工作区保留了冻结清单：

```bash
cd "${VLMGP_ROOT}"
shasum -a 256 \
  HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_train.json \
  HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_val.json \
  HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_test.json
```

全新公开克隆中这些清单被忽略，必须从发布的复现资产包恢复，或使用仓库工具重新构建后把新清单视为一次新的实验合同；不能假装其身份与上表完全相同。

## 7. 获取固定上游源码与模型权重

### 7.1 第三方源码

先阅读 `THIRD_PARTY_NOTICES.md`，再执行固定提交抓取脚本：

```bash
cd "${VLMGP_ROOT}"
sed -n '1,260p' THIRD_PARTY_NOTICES.md
bash scripts/fetch_external_repositories.sh
```

下表合并列出仓库内已有的固定源码快照和全新克隆时需要的依赖身份。`fetch_external_repositories.sh` 只会克隆 HiFi-CS、GQ-CNN、GraspNet API/baseline 与 VL-Grasp；CROG、GR-ConvNet、GG-CNN、LAVT 和 VGN 已作为仓库内快照存在，OpenAI CLIP 不由该脚本抓取，其固定身份另见 `configs/graspnet6d/environment.lock.txt` 和冻结环境证据。脚本会联网并写入缺失目录，但不会覆盖已有目录。关键固定提交：

| 组件 | 固定提交 | 许可证/使用边界 |
| --- | --- | --- |
| CROG | `1eeee85de1fe6bffdc66c9ed9a622028ea04578e` | MIT；源码已含本地适配 |
| OpenAI CLIP | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | 源码 MIT；预训练权重没有单独、清晰的权重许可证声明 |
| HiFi-CS | `4be6b3be7ce79fae481fb51616adfa2b803f07a0` | 审计时无仓库级许可证；只作参考/本地研究，不再分发 |
| GR-ConvNet | `bdd49367f8619be94123fb3187c2f8ad5100ef46` | BSD-3-Clause |
| GG-CNN | `0c50aa7600e8a30d44c5c85cebd6e3394a81f30e` | BSD-3-Clause |
| GQ-CNN | `499a609fe9dfb074bdfb6c4e6e33667ea50f4c21` | 研究/教育/非营利自定义条款 |
| LAVT-RIS | `1da0af9f21b637c0cae9ea1363d2dd9b40e19628` | GPL-3.0，隔离子树 |
| VGN | `d7af0622433f52ae88ebe81533f12b46b33e951a` | BSD-3-Clause；可选 6-DoF |

核对抓取结果：

```bash
git -C HiFi_reproduction/hifics rev-parse HEAD
git -C HiFi_reproduction/third_party/gqcnn-official rev-parse HEAD
```

### 7.2 权重来源、路径与哈希

只从官方发布地址下载。PyTorch 的 `.pth/.pt/.bin` 往往基于 pickle，视为可执行的不可信输入；先校验哈希，再在隔离环境加载。

| 权重 | 官方来源 | 本论文期望 SHA-256/说明 |
| --- | --- | --- |
| OpenAI CLIP RN50 | [RN50.pt](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) | `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762` |
| OpenAI CLIP ViT-B/16 | [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| GR-ConvNet Cornell | [固定提交中的官方文件](https://github.com/skumra/robotic-grasping/blob/bdd49367f8619be94123fb3187c2f8ad5100ef46/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98) | 7,640,481 bytes；正式初始化文件哈希 `e6c03afc0f8266d29ec92141088cc6a557215fabc2681d4ca5a162f451853d28` |
| GG-CNN2 Cornell | [v0.1 release ZIP](https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn2_weights_cornell.zip) | ZIP `f71e3575fe70bea6817239f9fad98264af5cfd95dd9bbe69ff4144696f34f972`；解包 state dict `865d538a51d427f7ee84defc99e093bdf51eeb0627c302068037e52188c11d1c` |
| GQ-CNN 2.1 | 官方模型下载由仓库脚本处理 | model-zoo 归档 `c3823f3525df851ea0b75c202e96e131ed85bafa9d4f3c4daa270a3c80943472`；`config.json` `eb5bc17089a39bd8fe6c801010c25a6a79a898d64181180feb5cf69aa630ff6f`；规范 22 文件 manifest `8201961abe3a09d90c6c66e582a3bfeb181d7095a2ebcc3a9d90e68fc12e8614` |

GR-ConvNet 和 GG-CNN 的 BSD-3-Clause 结论针对上游源码；上游没有为预训练权重另行给出清晰许可证。下载可用于本地复现，但再分发权重前应另行确认权利。GQ-CNN 下载器先尝试固定官方仓库的历史脚本；旧 Box 链接失效时，再使用官方 Dex-Net 文档指向的 Drive model-zoo 回退。

下载 CLIP 并严格校验：

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

下载目录只是统一保存点；两套 loader 实际读取不同固定位置。确认目标不存在后建立链接：

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

CROG Mac 配置读取 `exp/pretrain_clip/RN50.pt`；HiFi 训练器读取 `~/.cache/clip/ViT-B-16.pt`。若目标已经存在，先核验哈希，不要覆盖。

GR-ConvNet 正式初始化文件应放在：

```text
HiFi_reproduction/third_party_src/grconvnet/trained-models/
└── cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98
```

本论文选择的是 Cornell 权重，不是同目录可能存在的 Jacquard 权重。GG-CNN2 state dict 应放在：

```text
HiFi_reproduction/third_party_src/checkpoints/ggcnn2/
└── ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt
```

GG-CNN2 可以从官方 release 执行式下载：

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

GR-ConvNet 的固定上游提交直接跟踪该 7,640,481-byte Cornell 权重；使用 commit-pinned Raw URL，不使用漂移的 `main` 或第三方镜像：

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

该下载可恢复 G1/G0 的正式初始化权重身份；但原 G1/C1 从空目录开始的完整初始化命令账本没有保留，因此新训练仍应声明为按现有工具合同建立的新独立实验，而不是原运行逐命令重放。

本机正式微调后权重哈希：

```text
G1 best_state_dict.pt  5fc8cdae2578a2361c80d19d44ebf0df521938cd4c00c04fedb986b1f5169aea
C1 best_state_dict.pt  13addaa29f1f108888946b50467731b6fb53e34d4dcea5ba9d1d741a95147bc9
```

这些微调权重和 CROG 正式权重没有被 Git 跟踪；全新克隆要么从资产发布包取得并核验，要么按后文重新训练。

## 8. 为不同路线建立隔离环境

### 8.1 主 4-DoF/统一重排序环境

macOS/MPS 推荐 Python 3.11：

```bash
cd "${VLMGP_ROOT}"
uv venv --python 3.11 HiFi_reproduction/.venv-grasp4dof
uv pip install --python HiFi_reproduction/.venv-grasp4dof/bin/python \
  -r HiFi_reproduction/requirements-grasp4dof-macos.txt

export PY4="${VLMGP_ROOT}/HiFi_reproduction/.venv-grasp4dof/bin/python"
"${PY4}" -c 'import torch; print(torch.__version__); print("MPS", torch.backends.mps.is_available())'
PYTHONPATH=src "${PY4}" -m pytest -q tests/unified_reranking/test_metrics.py
```

若没有 `uv`，可用 `python3.11 -m venv` 后在该环境内 `pip install -r ...`。不要无理由升级依赖；优先保留正式运行中的 `package_lock.txt` 作为对照。

### 8.2 HiFi-CS repeated-FiLM 环境

HiFi-CS 上游没有稳定的锁文件；本论文运行依赖本地修改和运行时源码快照。当前本机可先核对：

```bash
test -f HiFi_reproduction/hifics/tools/train_hierfilm.py
test -f HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml
test -f HiFi_reproduction/artifacts/data_audit/frozen_manifests/ocidvlg_unique_train.json
```

若任一失败，公开克隆不具备精确训练前提。不要从上游 `main` 猜补文件；应恢复第 17 节的源码快照、配置和 manifest。环境创建方式：

```bash
python3.11 -m venv "${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics"
"${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics/bin/python" -m pip install --upgrade pip
"${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics/bin/python" -m pip install \
  -r "${VLMGP_ROOT}/HiFi_reproduction/hifics/requirements-macos.txt"
```

### 8.3 CROG 环境

官方 CUDA 依赖以 CROG 的 `environment.yml` 为准；本仓库 macOS 适配以 `README_MAC.md` 和 `requirements_mac.txt` 为准：

```bash
cd "${VLMGP_ROOT}/crog_reproduction/CROG"
python3.11 -m venv .venv-crog
.venv-crog/bin/python -m pip install --upgrade pip
.venv-crog/bin/python -m pip install -r requirements_mac.txt
.venv-crog/bin/python scripts/check_mps.py
```

如果官方依赖要求不同 Python 版本，应创建独立环境，不要把它并入 `.venv-grasp4dof`。

### 8.4 GQ-CNN/D1 Docker 环境

上游 GQ-CNN 1.3.0 支持 Python 3.5–3.7 和 TensorFlow `<=1.15.0`；本仓库的隔离评分容器固定为 Python 3.7.17/TensorFlow 1.15.0。本仓库提供固定 Dockerfile：

```bash
cd "${VLMGP_ROOT}"
docker info
docker build --platform linux/amd64 \
  -t vlmgrasp/gqcnn-score:1.3.0 \
  -f HiFi_reproduction/docker/gqcnn-score/Dockerfile \
  HiFi_reproduction
```

`docker info` 失败说明 daemon 未运行。启动 Docker Desktop 后再继续。不要把仓库可写挂载给未经核验的镜像；正式评分时仓库源码/输入只读挂载，模型目录按脚本要求单独可写。

## 9. 先做最小烟雾测试，而不是直接训练数小时

### 9.1 次级 target-aware demo

该 demo 能验证数据加载、目标掩码、RGB-D 几何候选、重排序和输出格式，但**不是论文正式模型**。

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

先用 `oracle` 隔离抓取侧问题；只有它通过后，才安装 `requirements-vlm.txt` 并尝试 Florence-2 + SAM。VLM 模式是当前 demo，不应写入论文正式结果表。

### 9.2 正式工具链烟雾测试

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

以上是数秒级 smoke。正式锁生命周期和独立重算测试会构造较大的合成表，在本机可能需要数分钟；准备启动正式 Test 前再运行：

```bash
PYTHONPATH=src "${PY4}" -m pytest -q \
  tests/unified_reranking/test_formal_test_orchestration.py::test_formal_lock_is_complete_immutable_and_hash_verified \
  tests/unified_reranking/test_formal_test_orchestration.py::test_formal_test_claims_before_single_label_read_and_finalizes_once
```

只有数据身份、权重哈希、测试和 smoke 都通过，才进入完整训练。

## 10. 阶段 A：复现 HiFi-CS repeated-FiLM 指代分割

### 10.1 这一步产出什么

HiFi-CS 只负责将语言表达与 RGB 图像映射为目标掩码，它本身不生成抓取姿态。论文使用的是本地实现的五级 hierarchical repeated-FiLM 变体，在 OCID-VLG `unique` 上受控重训。运行配置明确写着：

```text
controlled official-unique retraining; not an exact Table 2 reproduction
```

所以它复现的是本论文所用分割前端，不是对 HiFi-CS 论文 Table 2 的逐项精确复制。

### 10.2 核对冻结训练合同

源运行目录：

```text
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/
```

先检查其证据，不要启动训练：

```bash
cd "${VLMGP_ROOT}"
export HIFI_SOURCE="${VLMGP_ROOT}/HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615"

test -f "${HIFI_SOURCE}/COMPLETE"
jq '{mode, device, config_sha256, source_snapshot_sha256, git, manifests}' \
  "${HIFI_SOURCE}/startup_evidence.json"

# 按训练器写出的逐文件清单核验；不要对 source_snapshot/* 直接 shasum，
# 其中可能有 __pycache__/ 目录。
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

关键合同：

| 项 | 值 |
| --- | --- |
| 输入分辨率 | 352×352 |
| CLIP | ViT-B/16，图像/文本编码器冻结 |
| repeated-FiLM 层 | 5 级，投影层 1/3/5/7/9，解码顺序 9/7/5/3/1 |
| 优化器 | Adam，LR 0.001，weight decay 0 |
| batch / 累积 | 16 / 1 |
| 步数 | 最大 20,000 optimizer steps |
| 调度器 | cosine，`T_max=20000`，`eta_min=0.0001` |
| seed / dtype | 42 / float32 |
| 正式本机 device | MPS，AMP 关闭，workers 0 |
| 选择规则 | 最大 Validation foreground mIoU |
| Test 使用 | 最佳 Validation checkpoint 冻结后一次 |

部分超参数未在 HiFi-CS 论文中披露，是已记录的 released-code/controlled-run 选择；不要把它们误称为原论文规定值。

### 10.3 新运行配置

当前本机应存在：

```text
HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml
```

复制成专用配置，只改绝对路径和新输出目录，不改算法字段：

```bash
export HIFI_ID="hifics_ocidvlg_hierfilm_repro_$(date -u +%Y%m%dT%H%M%SZ)"
export HIFI_RUN="${VLMGP_ROOT}/HiFi_reproduction/runs/${HIFI_ID}"
export HIFI_CONFIG="${VLMGP_ROOT}/HiFi_reproduction/configs/${HIFI_ID}.yaml"

cp "${VLMGP_ROOT}/HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml" \
  "${HIFI_CONFIG}"
```

在 `HIFI_CONFIG` 中确认：

- `dataset_root` 指向 `${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG`；
- train/val/test manifest 指向冻结清单；
- `output_directory_pattern` 或 CLI `--run-dir` 指向新 `HIFI_RUN`；
- `resume: false`、`checkpoint: null`；
- 首次正式运行前不要打开 Test 标签或 Test 评估文件。

记录配置哈希：

```bash
shasum -a 256 "${HIFI_CONFIG}"
```

### 10.4 过拟合/小样本检查

训练器只支持 `full` 和 `overfit`。先做受限 overfit，不要把它当正式结果：

```bash
cd "${VLMGP_ROOT}/HiFi_reproduction/hifics"
export PYH="${VLMGP_ROOT}/HiFi_reproduction/.venv-hifics/bin/python"

"${PYH}" tools/train_hierfilm.py "${HIFI_CONFIG}" \
  --mode overfit \
  --run-dir "${HIFI_RUN}_overfit" \
  --max-steps 200 \
  --limit-samples 32
```

训练器对 overfit gate 强制要求正好 32 个样本、200–500 optimizer steps；其他值会主动报错。检查 loss 有限、参数确实变化、checkpoint 能读回、掩码语义没有反转。正式源运行保留了 `parameter_max_abs_changes.json`、`optimizer_coverage.json` 和定性审计，可作为对照。

### 10.5 完整训练与独立核验

```bash
cd "${VLMGP_ROOT}/HiFi_reproduction/hifics"

"${PYH}" tools/train_hierfilm.py "${HIFI_CONFIG}" \
  --mode full \
  --run-dir "${HIFI_RUN}"

"${PYH}" tools/verify_hierfilm_training.py "${HIFI_RUN}"
"${PYH}" tools/evaluate_hierfilm.py "${HIFI_RUN}" --device mps
"${PYH}" tools/verify_hierfilm_evaluation.py "${HIFI_RUN}"
```

`train_hierfilm.py` 本身只完成训练和 Validation，明确不在训练循环中加载 Test；`evaluation/` 与两个独立核验文件由后三条命令生成。到 `verify_hierfilm_evaluation.py` 通过为止，已经完成新 repeated-FiLM run 自身的训练、一次 HiFi 路线内 Test 和独立验证。

`report_hierfilm_results.py` **不是**新 run 的独立 finalizer。它还会读取 structural test 和旧 single-FiLM 基线；只有确实要重建这张历史对比报告时，才在以下三个前置资产全部存在后运行：

```text
HiFi_reproduction/artifacts/experiment/hierarchical_film_tests.json
HiFi_reproduction/runs/hifics_ocidvlg_20260711_112921/evaluation/evaluation_metrics.json
HiFi_reproduction/reports/hifics_previous_checkpoint_recomputed_protocols.json
```

第一项可由当前源码生成；后两项要求恢复旧 single-FiLM run，其中第三项通过重新评估旧 checkpoint 生成：

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

`evaluate_previous_fixed_protocols.py` 会拒绝覆盖已有输出。当前审计工作区缺少旧 run 的 `evaluation_metrics.json` 和既有重算文件，因此这张**旧基线比较报告**当前不可从现状直接重建；它不阻塞新 repeated-FiLM run 自身的完成标准。

正式源运行使用的原始命令记录在：

```text
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/command.json
```

完成标准：

```bash
test -f "${HIFI_RUN}/COMPLETE"
test -f "${HIFI_RUN}/evaluation/EVALUATION_COMPLETE.json"
jq . "${HIFI_RUN}/run_summary.json"
jq . "${HIFI_RUN}/evaluation/independent_metric_verification.json"
shasum -a 256 "${HIFI_RUN}/checkpoints/best.pth"
```

源运行最佳 checkpoint：epoch 12 / step 19,728，Validation foreground mIoU 约 0.820741；其 SHA-256 为：

```text
b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601
```

新的独立运行不要求得到完全相同哈希，但必须保留配置、源码快照、manifest、设备、随机种子、训练历史、最佳选择规则和独立重算。

## 11. 阶段 B：复现 G1/C1 模块化 4-DoF 后端

### 11.1 路线定义

- `R0`：repeated-FiLM 分割参考，不是抓取后端；
- `G0`：GR-ConvNet 预训练迁移；
- `G1`：GR-ConvNet 在 OCID-VLG 上微调；
- `C0`：GG-CNN2 预训练迁移；
- `C1`：GG-CNN2 在 OCID-VLG 上微调；
- `A0`：分析/几何基线；
- `*-O`：oracle-mask 诊断，仅用于定位分割瓶颈。

论文统一公平主面板最终使用 G1 和 C1。源正式运行：

```text
HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500
```

### 11.2 源码和输入审计

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

这里有一个重要的操作边界：`audit_sources.py` 硬编码验证正式源 HiFi run、checkpoint SHA，以及两个紧凑输入来源：

```text
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615
HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528
HiFi_reproduction/runs/modular_reranking_repeatedfilm_v1_20260729_203147
```

因此第 10 节训练出的新 `HIFI_RUN` 不会被第 11 节自动消费。复现论文锁定管线时应使用上述正式 HiFi checkpoint；若要把新 HiFi checkpoint 接入 G1/C1，必须修改 source contract、重新生成 compact inputs、重新审计并作为一个新的实验，不能沿用论文正式哈希或结果名称。

### 11.3 为什么不能给一个虚构的“一键命令”

此子系统没有一个已验证的总入口能够从空目录安全执行完所有阶段。正确的源事实是：

- 阶段工具在 `HiFi_reproduction/tools/grasp4dof/`；
- corrected 源运行的 `commands.log` 只有 20 行，从 `recompute_reference` 和正式 Test 尾段开始；
- 父运行 `modular_repeatedfilm_4dof_backends_v1_20260803_092055/commands.log` 只有 11 行，保留 G1/C1 training grid、部分 Validation 和选择命令；
- 最初的 audit、manifest、transfer tuning 和 lock 创建命令**没有完整账本**；manifest/锁能验证结果，但不能还原每一条初始化命令；
- 两个 `commands.log` 都包含源机器绝对路径，不能逐行盲执行；
- Test 阶段已经锁定，旧 run 不允许再次运行或覆盖。

先把两个不完整账本转换为可读清单：

```bash
sed -n '1,240p' "${MOD_SOURCE}/commands.log"
sed -n '1,240p' \
  "${VLMGP_ROOT}/HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_20260803_092055/commands.log"
```

创建新的模块化运行目录，并在每一步使用 `--help` 核对当前 CLI：

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

### 11.4 意图级阶段顺序（不是已验证的一键重放）

从代码依赖和冻结 manifest 可以确认以下顺序，但步骤 1–4 的完整原始 CLI 未保留。若要建立新的独立实验，应逐个读取当前 `--help`，先运行 `--dry-run`（若支持），并把实际命令写入新的账本；不要宣称这是原运行的逐命令重放：

1. `audit_sources.py`：固定 HiFi、GR-ConvNet、GG-CNN2 源码和预训练权重哈希；
2. `build_manifests.py`：构建 Train 26,295 / Validation 3,778 / Test 7,675 清单；
3. `select_development_subsets.py`：只从 Train/Validation 构造 smoke/pilot；
4. `tune_transfer_backends.py`：只在 Validation 调整 G0/C0 迁移参数；
5. `run_training_grid.py`：在 Train 微调 G1/C1，以 Validation 选 checkpoint；
6. `select_validation_primary.py`：冻结主方法和配置；
7. `build_formal_lock_config.py`：生成候选锁配置；
8. `lock_experiment.py`：锁定代码、数据、模型、配置、评估器；
9. 只在锁后执行正式 Test：`recompute_reference.py` 单独生成 R0；`run_method.py --split test` 只支持 G0/G1/C0/C1/A0；
10. `consolidate_results.py`、统计和 `independent_recompute.py`：聚合并独立重算；
11. 生成报告/图库并标记完成。

在阶段 8 之前，Test 只允许持有不含标签的样本身份；不得读取 `test_labels.parquet` 进行配置选择。

如果目标是复现论文**核心重排序结论**而不是重训抓取后端，最可靠的当前路径是：把 corrected source run 作为不可变 G1/C1 来源，通过第 13 节创建新的 unified run。若目标是 G1/C1 从零完整重训，当前状态应报告为 `PARTIALLY REPRODUCED`，直到发布初始化驱动脚本或完整命令账本。

### 11.5 G1/C1 的锁定选择值

| 参数 | G1 | C1 |
| --- | ---: | ---: |
| 输入尺寸 | 224 | 300 |
| mask 膨胀比例 | 0.10 | 0.15 |
| quality threshold | 0.10 | 0.05 |
| 最小 peak 距离 | 15 px | 20 px |
| raw 候选上限 | 100 | 100 |
| 固定矩形高度 | 20 px | 20 px |
| seed | 20260803 | 20260803 |
| LR / weight decay | 1e-4 / 1e-5 | 1e-4 / 0 |
| 最大 epoch / patience | 30 / 5 | 30 / 5 |
| 最佳 epoch | 23 | 30 |

所有值以 `selected_configs/G1.json` 和 `C1.json` 为机器可读源，表格只是帮助阅读。

### 11.6 重要口径冲突

模块化源报告中的 G1/C1 约 0.87/0.79 等数值来自早期“修正后端实验”的原生候选合同。统一公平运行重新定义并冻结了候选解码和同一评估合同，其正式 native 值是 0.475179/0.438176。两个实验的问题和候选合同不同，不能把差异解释成代码回归或重排序效果。

## 12. 阶段 C：复现 CROG 路线

CROG 是官方端到端 CLIP-based referring grasp synthesis 模型，论文与代码分别见 [PMLR 论文页](https://proceedings.mlr.press/v229/tziafas23a.html) 和 [官方仓库](https://github.com/HilbertXu/CROG)。官方工作定义的是 OCID-VLG 的 4-DoF referring grasp synthesis。

### 12.1 数据与 CLIP

```bash
cd "${VLMGP_ROOT}/crog_reproduction/CROG"
export PYC="${VLMGP_ROOT}/crog_reproduction/CROG/.venv-crog/bin/python"

"${PYC}" scripts/inspect_ocid_vlg.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml
```

确保配置的数据根指向同一个 OCID-VLG，CLIP RN50 的哈希为第 7 节值。

### 12.2 debug 训练与评估

```bash
"${PYC}" train_crog_mac.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml

"${PYC}" test_crog_mac.py \
  --config config/OCID-VLG/CROG_mac_mps_debug.yaml \
  --checkpoint exp/OCID-VLG_multiple_mac/CROG_mac_mps_debug/best_jindex_model.pth \
  --split val
```

debug 必须证明：批次能读、loss 有限、梯度更新、checkpoint 可加载、抓取矩形坐标与角度约定匹配。

### 12.3 正式 MPS 训练

本地配置：

```text
config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml
```

关键参数：输入 416、50 epochs、batch 8、Validation batch 24、LR 1e-4、milestone 35、seed 0。**当前配置可能保留旧 `resume` 路径**；复制配置并设置 `resume: null`、新输出目录后再开始，不能覆盖旧训练。

```bash
export CROG_ID="CROG_mps_repro_$(date -u +%Y%m%dT%H%M%SZ)"
export CROG_CONFIG="config/OCID-VLG/${CROG_ID}.yaml"
export CROG_OUTPUT="${VLMGP_ROOT}/crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/${CROG_ID}"
cp config/OCID-VLG/CROG_mac_mps_official_params_50epoch_bs8.yaml "${CROG_CONFIG}"

# 手动把 YAML 中 DATA.root_path 指向数据；把 TRAIN.exp_name 改成
# CROG_ID 的实际值；核对 TRAIN.output_folder；并设置 TRAIN.resume: null。
test ! -e "${CROG_OUTPUT}"
rg -n 'root_path:|exp_name:|output_folder:|resume:' "${CROG_CONFIG}"
"${PYC}" train_crog_mac.py --config "${CROG_CONFIG}"
```

注意 YAML 不会展开字面量 `${CROG_ID}`，必须把 `TRAIN.exp_name` 写成 shell 输出的实际 ID。`train_crog_mac.py` 的解析输出目录是 `TRAIN.output_folder/TRAIN.exp_name` 且允许目录已存在；若只复制配置、不改 `exp_name`，会继续写入旧正式目录。上面的 `test ! -e` 是启动前硬保护。

本机正式源 checkpoint：

```text
crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/
└── CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth
```

SHA-256：

```text
ac1304da520fbd3f6998dea88ba1e63b39b596b775856b39e5360a29576f1ddf
```

该 checkpoint 是本地按 `multiple` train/validation 协议训练并由 validation 选择的模型；统一公平运行随后把它冻结并在共同的 `unique` Test 7,675 样本上导出候选。它不是官方可下载 checkpoint，也不是在 `unique` Train 上重新训练的 CROG。

### 12.4 官方 CUDA 路线

官方仓库 README 报告 2×RTX 4090 训练约 3.5 小时；这是上游参考，不是本机 MPS 的时间保证。官方配置文件在本地为小写：

```bash
python -u train_crog.py \
  --config config/OCID-VLG/crog_multiple_r50.yaml
```

如果目标是本论文正式 `unique` 公平面板，必须使用对应冻结 split/候选合同；直接跑上游 `multiple` 配置只复现 CROG 上游基线，不会自动复现本论文结果。

### 12.5 导出公平候选

CROG 原生评估完成后，候选导出/修正工具位于 `crog_reproduction/CROG/failure_analysis/`，最终公平生成由统一入口的 P1 调用。不要先用 Test 标签手工挑选 CROG 阈值；统一运行会把候选几何、native score 和 evaluator 哈希写入合同。

但第 13 节的论文重放显式绑定已有 `fair_crog_hifics_g1_c1_no_rerank_...` 和 `old-feature-dir`，不会自动发现第 12 节刚训练的新 CROG checkpoint。使用论文锁定输入可复现原重排序结论；要评估新 CROG，必须先用 failure-analysis/export 工具为 Train/Validation/Test 生成新的候选来源、更新 source audit/old-feature 输入并建立全新的公平实验合同，不能沿用论文正式哈希。

## 13. 阶段 D：统一公平候选重排序 P0–P15

这是论文的核心实验。唯一权威操作说明是：

```text
runs/fair_unified_reranking_20260809_103012/README_REPRODUCE.md
```

### 13.1 两条铁律

1. `run_all --resume` **只负责 P0–P1** 的审计、paired manifests 和冻结候选池，不会跑完整实验。
2. 每次只执行 `_PIPELINE_STATUS.json` 中第一条可执行的 `next_commands`；不要跨阶段，更不要在旧锁定 run 上再次执行 Test。

### 13.2 初始化新 run

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

如果 `fair-run`、`modular-run` 或旧特征目录不存在/哈希不符，P0 应失败关闭。不要通过删掉审计或改哈希来继续；恢复正确来源资产。

`prepare_development.py`、`run_attribution_bridge.py` 和 `prepare_test_bridge_bundle.py` 还依赖上面的固定 G1/C1 bridge source（fold assignments、历史候选和 inventory），且部分路径在代码中硬编码。新 HiFi/G1/C1 run 不会自动替换它；要改变来源必须建立新的 bridge 合同并重新锁定。

### 13.3 P0–P1：审计、paired manifests、候选冻结

预期输出包括：

```text
00_audit/fair_source_audit.json
01_manifests/
02_candidates/candidate_contract_hashes.json
02_candidates/native_work/{crog,g1,c1}_{train,validation,test}/
commands.log
```

核对：三路线样本身份相同；G1/C1 使用同一 HiFi mask；native score 未被改写；候选几何成员关系固定；空输出数量可追踪。

### 13.4 P2–P5：开发标签、grouped folds 与特征

按 status 给出的阶段依次执行。这些阶段必须满足：

- 标签只来自 Train/Validation；
- 同一场景/帧不能跨训练与验证 fold；
- 候选标签使用第 1.2 节同一评估器；
- 特征不得含 GT mask IoU、失败分类或任何 Test 标签派生量；
- 每条候选保留 route、sample、rank、geometry、native score 和来源哈希。

每次执行后：

```bash
"${PY4}" -m tools.unified_reranking.pipeline_status \
  --run-dir "${UNIFIED_RUN}" --write-status
jq . "${UNIFIED_RUN}/_PIPELINE_STATUS.json"
```

`next_commands` 中某些 P3/P4/P8/P9 项是模板，可能含 `<route>`、`<split>` 或 `...`，不能把占位符原样粘贴到 shell。用正式源 ledger 查看该 stage/substage 的具体展开，再把根目录和新 run 路径参数化；不要盲目批量执行 ledger：

```bash
export STAGE_TO_EXPAND="P3_P4"
jq -r --arg stage "${STAGE_TO_EXPAND}" \
  'select(.stage==$stage and .status=="COMPLETE") |
   [.substage,.command,.artifact_path] | @tsv' \
  "${VLMGP_ROOT}/runs/fair_unified_reranking_20260809_103012/commands.log"
```

对每个展开命令先核对 `--help`、输入哈希和输出目录，再执行到新的 `UNIFIED_RUN`。若 source ledger 没有同一 substage，停止并记录 blocker，不要猜参数。

### 13.5 P6–P10：校准、模型矩阵、Validation 选择、门控/路由

仍然只运行 `next_commands`。本阶段执行：

- P6：只用开发集拟合/校准候选分数；
- P7：运行预注册模型、特征、loss、seed 矩阵；
- P8：按 Validation 指标筛选主 ranker；
- P9：选择保守 gate、route router/union；
- P10：生成 attribution bridge，但不重新选择 Test 最优配置。

“保守门控”不是事后挑选好案例，而是 Validation 冻结策略：只有选择器信心足够时才从 native Top-1 切换，否则保留 native。

### 13.6 P11：预锁与正式锁

只有 status 报告 P11 ready 时，执行：

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

然后检查：

```bash
test -f "${UNIFIED_RUN}/08_lock/FORMAL_TEST_LOCK.json"
jq . "${UNIFIED_RUN}/08_lock/FORMAL_TEST_LOCK.json"
"${PY4}" -m tools.unified_reranking.pipeline_status \
  --run-dir "${UNIFIED_RUN}" --json \
  | jq -e '.first_actionable_stage=="P12" and .stages.P12.status=="READY"'
```

不要用 `--require-ready P12` 做“尚未完成但已 READY”的预检：当前实现会把缺少 P12 输出视为失败。上面的 JSON 断言才检查“下一步正好是 P12”。

### 13.7 P12：Test 正式执行一次

这是不可逆的科学流程节点。开始前备份锁文件和哈希；确认没有任何人/进程已在此 run 执行过：

```bash
test ! -f "${UNIFIED_RUN}/09_formal_test/FORMAL_TEST_EXECUTION.json"
```

只有前述检查通过才执行：

```bash
"${PY4}" -m tools.unified_reranking.run_formal_test_once \
  --run-dir "${UNIFIED_RUN}"
```

如果命令中断，不要删除 execution marker 后重跑；先查看 ledger/status 的恢复语义。工具设计为 exactly-once，绕过保护会破坏论文有效性。

### 13.8 P15 与 post-formal 独立重算

```bash
"${PY4}" -m tools.unified_reranking.independent_recompute \
  --run-dir "${UNIFIED_RUN}"

"${PY4}" -m tools.unified_reranking.build_postformal_artifacts \
  --run-dir "${UNIFIED_RUN}"

"${PY4}" -m tools.unified_reranking.pipeline_status \
  --run-dir "${UNIFIED_RUN}" --write-status
```

完成时至少应满足：

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

### 13.9 使用 commands.log 追溯每一步

源正式运行有 1,814 条 JSONL 记录。它是完整命令/产物/哈希账本，不需要在本指南复制 1,814 行：

```bash
export UNIFIED_SOURCE="${VLMGP_ROOT}/runs/fair_unified_reranking_20260809_103012"

jq -r 'select(.status=="COMPLETE") |
  [.stage,.substage,.command,.artifact_path,.artifact_sha256] | @tsv' \
  "${UNIFIED_SOURCE}/commands.log" \
  > "${REPRO_ROOT}/formal_command_ledger.tsv"

awk -F '\t' '{count[$1]++} END {for (stage in count) print stage, count[stage]}' \
  "${REPRO_ROOT}/formal_command_ledger.tsv" | sort
```

旧命令带绝对路径，只能作为参数来源；新的 run 必须由当前 `pipeline_status` 驱动，不能直接 `bash commands.log`。

## 14. 阶段 E：D1（Dex-Net/GQ-CNN）单独复现

D1 不是语言模型，也不是 6-DoF pose generator。候选侧先用 HiFi 掩码限制目标区域，Dex-Net 生成几何候选，GQ-CNN 评分；另一个独立的 retrospective run 才对固定候选重排序。必须标注 `D1_COMPATIBILITY` 或 `D1_RETROSPECTIVE`，不能与 FORMAL_PRIMARY 混成一张无范围标签的表。

### 14.1 下载模型和构建容器

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

下载脚本依赖容器内 `/opt/gqcnn`，不能直接在 macOS 主机上裸执行。脚本本身不验证哈希；上面的命令验证 fallback 归档（若官方旧脚本直接成功则可能不存在该归档）、`config.json` 和文件数。D1 初始化还会按“相对路径、文件 SHA、字节数”的规范 JSON 计算 22 文件 manifest，并要求其 SHA-256 为 `8201961abe3a09d90c6c66e582a3bfeb181d7095a2ebcc3a9d90e68fc12e8614`。GQ-CNN 条款限制研究、教育和非营利用途，使用前自行复核当前上游许可。

精确源运行还锁定了 Docker image ID：

```text
sha256:3d1158ca83197d55808454b718d0a328d3f27c57c80baaaea7031e21a9134ebd
```

即便 Dockerfile 相同，重新构建也可能因基础镜像/包索引变化得到不同 image ID，D1 初始化会按设计拒绝。要做精确 D1，需恢复该镜像归档；若接受新镜像，则应创建并记录新的 D1 实验合同，不能称为比特级重放。

### 14.2 生成/评分兼容性来源（不含重排序）

编排器：

```text
HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py
```

它会硬校验数据 manifest、HiFi checkpoint、GQ-CNN 模型、Docker 镜像和输入图像。只对**新目录**初始化：

```bash
export D1_RUN="${VLMGP_ROOT}/HiFi_reproduction/runs/d1_repro_$(date -u +%Y%m%dT%H%M%SZ)"

"${PY4}" HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py \
  --run-dir "${D1_RUN}" \
  --initialize \
  --initialize-only
```

核对生成的 manifest/计划后执行：

```bash
"${PY4}" HiFi_reproduction/scripts/run_hierfilm_modular_experiment.py \
  --run-dir "${D1_RUN}" \
  --resume
```

不要对已存在目录再次加 `--initialize`。该编排器的预注册顺序是 mask 导出 → Dex-Net 候选 → GQ-CNN 评分 → corrected compatibility evaluation/报告。源协议明确禁止 reranker 或训练；它只生成 D1 的候选/分数来源，不会产生论文的 +18.619pp 重排序结果。

### 14.3 运行独立的 D1 retrospective 重排序

论文 D1 headline 的权威来源是：

```text
runs/fair_d1_reranking_extension_20260811T145515Z/
```

该 run 有 988 行 `commands.log`，协议名为 `fair-d1-reranking-retrospective-extension-v1`，明确声明它是回顾性扩展而非 pristine blind Test 实验。它绑定：

```text
runs/fair_unified_reranking_20260809_103012
HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528
canonical evaluator SHA-256 f5155590b0b8d9f0748ad463688edfe6ca595d8ab7239ef5e29a5c595aeef301
```

先验证静态 P0–P17 DAG 和来源：

```bash
cd "${VLMGP_ROOT}"
export D1_RETRO_SOURCE="${VLMGP_ROOT}/runs/fair_d1_reranking_extension_20260811T145515Z"

test -f "${D1_RETRO_SOURCE}/COMPLETE"
PYTHONPATH=src "${PY4}" -m tools.d1_reranking.render_command_dag \
  --check --format markdown
jq . "${D1_RETRO_SOURCE}/pipeline_status.json"
```

当前 `render_command_dag` 生成的 bootstrap 示例仍可能带有 `bootstrap` CLI 已不支持的 `--evaluator` 参数；DAG 只作为阶段顺序权威，不能复制其中的示例命令。实际初始化使用下面经当前 `--help` 核对的命令。

若要创建新 retrospective run：

```bash
export D1_RETRO_RUN="${VLMGP_ROOT}/runs/fair_d1_reranking_repro_$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONPATH=src "${PY4}" -m tools.d1_reranking.bootstrap \
  --run-dir "${D1_RETRO_RUN}" \
  --unified-run "${VLMGP_ROOT}/runs/fair_unified_reranking_20260809_103012" \
  --snapshot-a "${VLMGP_ROOT}/HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
```

之后以新 `pipeline_status.json` 的 `first_incomplete_stage` 和静态 DAG 为顺序权威，并从源 ledger 展开同一 stage 的具体命令。D1 没有 `pipeline_status` CLI 或已验证的一键 end-to-end driver；以下是逐阶段、需人工核对 `--help` 和输入哈希的恢复路径：

```bash
export D1_STAGE="P1"
jq -r --arg stage "${D1_STAGE}" \
  'select(.stage==$stage and .status=="COMPLETE") |
   [.substage,.command,.artifact_path,.artifact_sha256] | @tsv' \
  "${D1_RETRO_SOURCE}/commands.log"
```

不要把源绝对路径逐行盲执行，也不要向已完成源 run 写入。P0–P13 全部完成并达到锁前条件后，正式尾段的工具顺序为：

```bash
PYTHONPATH=src "${PY4}" -m tools.d1_reranking.assemble_formal_evaluation_plan \
  --run-dir "${D1_RETRO_RUN}" --resume

PYTHONPATH=src "${PY4}" -m tools.d1_reranking.create_formal_lock \
  --run-dir "${D1_RETRO_RUN}" \
  --evaluation-plan "${D1_RETRO_RUN}/configs/d1_formal_evaluation_plan.json"

# exactly once；执行前确认 pipeline_status/lock，不可对源 run 重跑。
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

中间 P1–P13 包含来源对账、候选冻结、开发标签、fold、特征、校准、主模型矩阵、gate、K 敏感性、消融和四路线扩展。它们必须由新 run 的资源授权/锁状态驱动；源 ledger 中还有失败重试和对账记录，不能简单地执行全部 988 行。

### 14.4 D1 的两个不可混用合同

- `D1_COMPATIBILITY`：来源原生的 GQ-CNN all-NMS 顺序与兼容评估器；
- `D1_RETROSPECTIVE`：冻结 post-NMS Top-5，使用与正式主面板相同的 evaluator SHA 进行回顾性重排序。

论文 +18.619 个百分点的 D1 数字属于 `D1_RETROSPECTIVE`。仓库中约 0.4627/0.7053/0.8377 的兼容性指标是另一合同，不能替换它。

## 15. 阶段 F：重建证据、图表和学位论文

### 15.1 只重建派生证据

源证据包：

```text
runs/four_route_evidence_consolidation_20260813T155455Z/
```

其 `10_reproducibility/README_REPRODUCE.md` 明确只允许从冻结输出重建派生资产；禁止训练、候选生成、阈值选择、正式 Test 和 D1 finalizer 写回源 run。

```bash
cd "${VLMGP_ROOT}"
export EVIDENCE_RUN="${VLMGP_ROOT}/runs/four_route_evidence_consolidation_20260813T155455Z"
export PYEVIDENCE="${VLMGP_ROOT}/HiFi_reproduction/.venv-grasp4dof/bin/python"

"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/build_bundle.py"
```

演示文稿构建依赖 Node 运行时。当前 Codex 工作区已验证的 bundled runtime 如下；若复制到其他机器，应改为该机器兼容的 Node，并把 `NODE_PATH` 指向含 `pptxgenjs` 等依赖的目录：

```bash
export CODEX_NODE="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
export NODE_PATH="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"

"${CODEX_NODE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/build_deck.mjs"

soffice --headless --convert-to pdf \
  --outdir "${EVIDENCE_RUN}/09_presentation" \
  "${EVIDENCE_RUN}/09_presentation/Four_Route_Evidence_Consolidation.pptx"
```

先完成 evidence 自带的论文片段编译检查；`verify_bundle.py` 明确要求其 PDF 存在。这一步不同于下一节的完整学位论文编译：

```bash
cd "${EVIDENCE_RUN}/08_thesis"
latexmk -lualatex -interaction=nonstopmode -halt-on-error \
  thesis_compile_check.tex
cd "${VLMGP_ROOT}"
```

随后执行快照与完整性验证：

```bash
"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/source_snapshot.py" after
"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/source_snapshot.py" compare
"${PYEVIDENCE}" "${EVIDENCE_RUN}/10_reproducibility/scripts/verify_bundle.py"
```

系统 `python3` 缺少构建依赖；`.venv-paper-tools` 虽有 `pandas`/`matplotlib`，当前没有 `pyarrow`，读取 Parquet 会失败。上面的 `.venv-grasp4dof` 已验证同时含 `pandas`、`matplotlib` 和 `pyarrow`，因此复用它运行证据脚本。

### 15.2 编译学位论文

论文模板使用 LuaLaTeX/Biber；不要改用 pdfLaTeX/BibTeX：

```bash
cd "${VLMGP_ROOT}/MSc_Dissertation_Template_the_University_of_Manchester_EEE__2025_onwards"

latexmk -g -lualatex -shell-escape \
  -interaction=nonstopmode \
  -halt-on-error \
  main.tex
```

成功标准：命令退出码 0、引用不再 undefined、机器可读表格与正文数值一致、生成 `main.pdf`。如果只检查证据片段，进入 evidence bundle 的 `08_thesis/` 按其 README 构建 `thesis_compile_check.tex`。

## 16. 如何判断复现成功

不要只比较一行平均数。按以下层级验收：

1. **身份**：Git 提交、第三方提交、配置、数据清单、样本顺序、checkpoint、评估器 SHA 均有记录；
2. **数据**：unique Train/Val/Test 分别 26,295/3,778/7,675，场景/帧无开发泄漏；
3. **生命周期**：Test 标签在锁前未访问，正式 Test execution count 为 1；
4. **候选合同**：CROG/G1/C1 各自 route-local Top-5，native score 和几何未被改写；
5. **指标**：同一 GT 同时满足 IoU/角度，空输出留在分母；
6. **方向性**：三条正式路线 gated ΔJ@1 均为正，恢复数明显多于伤害数；
7. **统计**：scene-cluster bootstrap 95% 下界 > 0、Holm 校正 McNemar p < 0.05；
8. **独立重算**：与正式产物逐项一致；
9. **报告**：表格、图、摘要和论文 PDF 从机器可读产物生成，无手抄漂移；
10. **声明边界**：只声称离线 4-DoF 代理改进，不声称物理成功率或完整 6-DoF 执行。

允许的结果报告格式示例：

```text
状态：COMPUTATIONALLY REPRODUCED / PARTIALLY REPRODUCED / NOT REPRODUCED
代码提交：...
数据 manifest set SHA-256：...
设备与软件：...
正式 Test execution count：1
路线：CROG/G1/C1
N：7,675
Native J@1：...
Gated J@1：...
ΔJ@1：...
Recovered/Harmful：...
独立重算：PASS/FAIL
与参考差异及原因：...
```

## 17. 全新公开克隆当前缺什么

截至本指南审计提交，下列核心路径的 `git ls-files` 结果为 0，且 `.gitignore` 排除了数据、模型、checkpoint、`*.pth/*.pt`、所有 `**/runs/`：

```text
runs/fair_unified_reranking_20260809_103012/
runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523/
runs/fair_d1_reranking_extension_20260811T145515Z/（D1 retrospective 的 988 行账本、锁和结果）
runs/reranking_case_visuals_v2_20260810T203154Z/（证据包案例图来源）
runs/reranking_complete_20260803_094159/features/（P1 的 CROG Train/Validation 冻结候选来源）
runs/four_route_evidence_consolidation_20260813T155455Z/
HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/
HiFi_reproduction/runs/hifics_ocidvlg_20260711_112921/（D1 还依赖旧 single-FiLM 模板/预测）
HiFi_reproduction/runs/grasp_backend_comparison_20260807_090155/（证据包 COMP_DIAG 来源）
HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/
HiFi_reproduction/runs/modular_reranking_repeatedfilm_v1_20260729_203147/
HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_20260803_092055/（G1/C1 training grid、manifest 与父账本来源）
HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/
HiFi_reproduction/runs/g1_c1_complete_reranking_20260806T084131Z/（fold 与 attribution/test bridge 来源）
HiFi_reproduction/hifics/
HiFi_reproduction/configs/hifics_ocidvlg_hierfilm_controlled.yaml
HiFi_reproduction/artifacts/data_audit/frozen_manifests/
HiFi_reproduction/artifacts/experiment/hierarchical_film_tests.json（旧基线比较报告需要，可由源码重建）
HiFi_reproduction/reports/hifics_previous_checkpoint_recomputed_protocols.json（旧基线比较报告需要）
HiFi_reproduction/third_party_src/grconvnet/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98（可由第 7.2 节官方固定 URL 恢复）
HiFi_reproduction/third_party_src/checkpoints/ggcnn2/ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt
HiFi_reproduction/models/gqcnn-official/
crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth
OCID-VLG 数据
```

要让其他研究者从公开仓库完成 R2/R3，建议发布一个带版本号和总 SHA-256 的复现资产包，至少包括：

1. 数据下载说明、原始 ZIP 自记录哈希和三份冻结 manifest（不一定再分发数据本体）；
2. HiFi 上游固定提交与下载脚本、项目自有 patch/diff、controlled config、环境锁、最佳 checkpoint 和 snapshot 哈希 manifest；完整上游 source snapshot 只保留为私有/本地审计资产，除非作者明确授予再分发许可；
3. GR-ConvNet/GG-CNN2 官方权重来源、完整哈希和 G1/C1 微调 checkpoint；
4. CROG 精确 config、环境锁、正式 checkpoint 和候选导出 manifest；
5. fair native source run、unified run 的锁/ledger/必要 Parquet/报告；
6. D1 GQ-CNN/Dex-Net 输入合同、模型哈希，以及锁定 Docker image 的可验证归档（若声明精确 D1 可复现）；
7. evidence bundle 和论文所引用图表；
8. 每个第三方资产的许可证/再分发权限。无许可证或不允许再分发的组件只发布下载脚本、固定提交和 patch，不打包源码/权重。

正式统一源运行记录的仓库基线曾是 `522b5f2156e7ee7bb02ac3aba535aa58ec890ca7` 加 `00_audit/current_git_diff.patch`，而本指南审计的公开 HEAD 是 `461505999123fe23303bab45bfc7ddbf9619f1fb`。精确结果的源码身份应以正式 lock/source audit 为准，不能只用当前 clean HEAD 替代。

## 18. 常见失败、原因和修复顺序

### 18.1 找不到 OCID-VLG

```bash
ls -ld "${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG"
ls "${VLMGP_ROOT}/HiFi_reproduction/OCID-VLG/refer/unique"
```

常见原因是多解压了一层或软链接指向不存在路径。先找真正包含 `ARID10/ARID20/refer` 的目录，再重建链接；不要复制三份。

### 18.2 manifest 数量或哈希不同

可能下载版本、相对路径标准化、样本排序或本地生成代码不同。停止正式训练，比较每条 record 和有序 sample ID；不能只把 expected hash 改成新值。

### 18.3 MPS 不可用或算子失败

先运行小样本和相关单元测试。若退回 CPU，记录 device 改变；若转 CUDA，建立新环境并报告数值差异。不要在训练中途无记录切设备。

### 18.4 `run_all --resume` 后没有完整结果

这是预期行为：它只到 P1。运行 `pipeline_status --write-status`，执行 `_PIPELINE_STATUS.json` 的第一条 `next_commands`。

### 18.5 formal Test 拒绝第二次运行

这是安全保护，不是 bug。换一个全新 run 从头执行；不要删除锁或 execution marker。

### 18.6 G1/C1 数字与旧报告差很多

先查候选池、selector、角度符号、NMS 和 evaluator contract。统一公平与历史模块化结果不可直接比较。

### 18.7 Docker/GQ-CNN 失败

确认 Docker daemon、`linux/amd64` 模拟、模型内外层哈希和只读挂载；原生 macOS 新版 Python 无法等价替代 TensorFlow 1.15 环境。

### 18.8 LaTeX 编译失败

确认使用 LuaLaTeX、安装 Biber/所需字体/包、加 `-shell-escape`，先编译最小 `thesis_compile_check.tex` 定位问题。不要手动修改生成表中的数字来绕过构建错误。

## 19. 可选扩展：不要与论文主复现混淆

### 19.1 Florence-2 + SAM demo

仅用于 `target_aware_vlm_grasping`。默认期望：

```text
target_aware_vlm_grasping/models/vlm/florence2-large-ft/
target_aware_vlm_grasping/models/vlm/sam/sam_vit_b_01ec64.pth
```

Florence-2 应固定 Hugging Face revision 并优先 safetensors；SAM ViT-B 使用 Meta 官方 checkpoint。涉及 `trust_remote_code=True` 时，必须固定 revision 并先审阅代码。

### 19.2 GraspNet/6-DoF

官方入口见 [GraspNet 数据页](https://graspnet.net/datasets.html)。仓库提供：

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python -m graspnet6d.cli all \
  --profile paper-lite \
  --run-id "graspnet6d_$(date -u +%Y%m%dT%H%M%SZ)"
```

中断后只对同一 run 使用文档声明的 `--resume`；全新实验必须换新 run ID。该项目采用 `src/` 布局且未安装为 site package，从 clean shell 省略 `PYTHONPATH=src` 会报 `ModuleNotFoundError`。

先阅读 `MANUAL_DOWNLOAD_REQUIRED.md` 和 `src/graspnet6d/compact_download.py`。当前本机的 30/3 场景子集只可做开发验证，不是 GraspNet 官方 benchmark，也不能支持实体成功率结论。

### 19.3 VGN/LAVT/VL-Grasp/AnyGrasp

它们可用于后续对照或三维扩展，但许可证、数据合同、坐标系和指标不同。新增实验必须建独立 scope、manifest、评估器和结果表，不能追加到 FORMAL_PRIMARY 行中。

## 20. 主张—证据—来源映射

| 主张 | 机器可读/论文证据 | 外部来源 |
| --- | --- | --- |
| OCID-VLG 是语言指代的 4-DoF 抓取数据 | 冻结 unique manifests、论文第 3/4 章 | [OCID-VLG 官方仓库](https://github.com/gtziafas/OCID-VLG) |
| CROG 是端到端 CLIP referring grasp synthesis | CROG config/checkpoint/candidate manifest | [CROG PMLR](https://proceedings.mlr.press/v229/tziafas23a.html)、[代码](https://github.com/HilbertXu/CROG) |
| HiFi-CS 在本论文中只提供 repeated-FiLM 目标 mask | HiFi source snapshot、config、run summary | [HiFi-CS 仓库](https://github.com/vineet2104/hifics)、[论文](https://arxiv.org/abs/2409.10419) |
| G1/C1 是共享 HiFi mask 的模块化后端 | G1/C1 selected configs、source audit | [GR-ConvNet](https://github.com/skumra/robotic-grasping)、[GG-CNN](https://github.com/dougsm/ggcnn) |
| 正式 Test 是 unique N=7,675、只执行一次 | FORMAL_TEST_LOCK、FORMAL_TEST_EXECUTION、ledger | OCID-VLG 官方划分说明 |
| 重排序提高三条正式路线 J@1 | final results、scene bootstrap、McNemar、independent recompute | 本仓库正式运行证据 |
| D1 需单列 | scope contract map、D1 manifests | [GQ-CNN](https://github.com/BerkeleyAutomation/gqcnn) |
| 离线矩形指标不等同实体成功率 | evaluator contract 与无 ROS/硬件执行证据 | CROG 原论文的机器人实验是另一实验系统 |

## 21. 外部资源与复用决定

本复现优先复用仓库内正式代码、锁和账本；外部实现只用于固定的上游模型/数据来源：

- OCID-VLG：使用作者数据与官方 split，不复制第三方镜像；
- CROG：MIT 源码固定提交，在仓库本地适配版上训练/导出；
- HiFi-CS：固定提交并参考作者实现，但因缺少仓库级许可证，不把其代码重新打包发布；本论文修改以项目自有 patch 管理，完整 source snapshot 仅作私有/本地审计，除非作者另行授予再分发许可；
- GR-ConvNet、GG-CNN/VGN：按 BSD-3-Clause 保留上游源码许可证和提交，权重再分发权另行确认；
- OpenAI CLIP：保留固定的 MIT 源码提交，不能从源码许可证推导出单独的预训练权重许可证；
- GQ-CNN：只按其研究/教育/非营利条款本地使用；
- 未发现一个可以合法、安全、无损替代本仓库 P0–P15 锁定重排序系统的外部实现，因此核心重排序使用本仓库自有代码，不从博客/Stack Overflow 复制代码。

## 22. 最终逐项清单

### 获取与身份

- [ ] 固定 VLMGraspPose 提交并保存 dirty diff
- [ ] 阅读 `THIRD_PARTY_NOTICES.md`
- [ ] 固定所有上游提交
- [ ] 下载 OCID-VLG 并保存自记录 ZIP SHA-256
- [ ] 核对 unique manifest 数量与哈希
- [ ] 下载权重并逐个校验完整 SHA-256

### 训练与候选

- [ ] HiFi overfit smoke 通过
- [ ] HiFi full 训练、Validation 选择、HiFi 路线内 Test once、独立核验完成
- [ ] G1/C1 source audit 和数据审计通过
- [ ] G1/C1 只用 Validation 选配置并锁定
- [ ] CROG debug/full、checkpoint 与候选导出完成
- [ ] 三路线 paired sample identity 与 Top-5 候选合同通过

### 重排序与正式测试

- [ ] P0–P10 由 `pipeline_status` 顺序驱动
- [ ] 没有 GT/Test 特征泄漏或场景/帧泄漏
- [ ] P11 预锁和 formal lock 完成
- [ ] P12 execution count 正好为 1
- [ ] P15 独立重算及全部完整性检查 PASS
- [ ] D1 若运行，按 compatibility/retrospective 单列

### 写作与报告

- [ ] 报告 FORMAL_PRIMARY 的 N、native、gated、增量、恢复/伤害和统计区间
- [ ] 不把旧 `multiple`、legacy evaluator 或 D1 与主表混合
- [ ] 不把 4-DoF 离线 J@1 写成 6-DoF/实体成功率
- [ ] evidence bundle 验证通过
- [ ] LuaLaTeX 学位论文编译通过
- [ ] 记录与参考值的所有差异和仍未复现的资产

---

如果目标只是验证代码能工作，执行第 5–9 节即可；如果目标是重新得到论文主数字，必须完成第 6–13、15–18 节；如果目标是让第三方从公开 GitHub 独立完整复现，还必须先完成第 17 节的资产发布。
