# Modular Candidate Re-ranking V1：审计报告

> 文档状态：截至 Run ID `20260728T210046+0100` 的可复算审计。  
> baseline、split、工程 smoke 和单元测试已完成；full validation、实验锁和
> formal test 仍为 `PENDING`。

## 1. 审计问题

本审计先回答一个比“新方法高不高”更重要的问题：

> 当前 failure 是模型的真实能力限制，还是 checkpoint、数据处理、坐标恢复、
> 候选排序或 evaluator 的复现错误？

审计范围包括：

- 冻结 HiFi-CS、Dex-Net、GQ-CNN 输出的数量和身份；
- NPZ 原候选顺序与 JSON/CSV q-rank 顺序的区别；
- grasp rectangle 的 `(x,y)` 坐标和 180° 夹爪周期角；
- `IoU > 0.25` 严格边界与历史 `IoU >= 0.25` 行为；
- train/validation/test 的 sample、scene、RGB、depth 和 RGB-D pair 泄漏；
- 新的 compact pipeline 与冻结 test 输出的一致性；
- inference allowlist、GT 后连接和 VLM prompt 隔离；
- local VLM 是否只在回环地址运行且关闭 cloud；
- 相关测试能否重复通过。

## 2. 冻结 test baseline 复现

机器可读来源：

`outputs/modular_reranking_v1/20260728T210046+0100/baseline_full/baseline_audit.json`

SHA-256：

`44b9603e2949a35ae1798a878b713d221a5df73e0fc9c9122563ca53293b96f9`

### 2.1 数量审计

| 项目 | 复算值 | 结论 |
|---|---:|---|
| 总样本 | 7,675 | 匹配 |
| 独立 RGB-D scene-frame | 325 | 匹配 |
| raw candidates | 1,567,552 | 匹配 |
| mask-validated candidates | 1,551,137 | 匹配 |
| NMS candidates | 206,538 | 匹配 |
| 非空样本 | 7,620 | 匹配 |
| 合法空样本 | 55 | 匹配 |
| finite GQ-CNN q | 206,538 | 匹配 |
| execution failures | 0 | 匹配 |
| 非空但 full NMS pool 无正候选 | 1,506 | 匹配 |

### 2.2 历史口径与严格论文口径

历史 evaluator 使用 `IoU >= 0.25`，可以精确复现：

| 历史口径 | 非空样本 | 全体样本 |
|---|---:|---:|
| Top-1 | 2,835 / 7,620 = 37.2047% | 2,835 / 7,675 = 36.9381% |
| Top-5 | 4,787 / 7,620 = 62.8215% | 4,787 / 7,675 = 62.3713% |
| Top-10 | 5,459 / 7,620 = 71.6404% | 5,459 / 7,675 = 71.1270% |
| MRR（无正候选记 0） | 0.4867738383 | 0.4832855567 |

论文规定的 canonical predicate 是 `IoU > 0.25`：

| 严格口径 | 非空样本 | 全体样本 |
|---|---:|---:|
| Top-1 | 2,833 / 7,620 = 37.1785% | 2,833 / 7,675 = 36.9121% |
| Top-5 | 4,787 / 7,620 = 62.8215% | 4,787 / 7,675 = 62.3713% |
| Top-10 | 5,459 / 7,620 = 71.6404% | 5,459 / 7,675 = 71.1270% |
| full NMS Oracle | 6,114 / 7,620 = 80.2362% | 6,114 / 7,675 = 79.6612% |
| MRR（无正候选记 0） | 0.4866426047 | 0.4831552636 |

14 个 NMS candidate 的标签因严格边界而改变，其中 2 个是历史 rank-1；
因此历史 Top-1 2,835 与严格 Top-1 2,833 都是可解释、可复现的数字，但不能
混写。后续新实验必须使用严格 `>`。

> **像给 12 岁孩子解释**
>
> 老规则说“考到 25 分也算过”，新规则说“必须高于 25 分”。正好有两道
> Top-1 答案卡在 25 分，所以历史答案是 2,835，新规则答案是 2,833。这不是
> 模型突然变差，而是及格线的符号从“≥”变成了“>”。

### 2.3 first-valid rank

在严格口径下，full NMS pool 有正候选的样本为 6,114：

- mean first-valid rank：4.3635917566；
- median first-valid rank：2；
- positive-sample MRR：0.6065123729。

历史 `>=` 口径的 mean first-valid rank 为 4.3632646385。

## 3. Oracle Funnel 与失败发生在哪一层

### 3.1 累积 Oracle

| 阶段 | 至少有一个正候选 | 全体样本比例 |
|---|---:|---:|
| raw | 6,288 | 81.9283% |
| mask validated | 6,276 | 81.7720% |
| NMS / full-pool Oracle | 6,114 | 79.6612% |
| GQ-CNN Top-10 | 5,459 | 71.1270% |
| GQ-CNN Top-5 | 4,787 | 62.3713% |
| GQ-CNN Top-1 | 2,833 | 36.9121% |

### 3.2 互斥 failure 类别

| 类别 | 样本数 | 含义 |
|---|---:|---|
| `generation_limited` | 1,333 | raw pool 已没有符合 GT 的候选 |
| `mask_filter_loss` | 11 | raw 有正候选，mask filtering 后丢失 |
| `nms_loss` | 162 | mask 后有正候选，NMS 后丢失 |
| `ranking_loss_beyond_top5` | 1,327 | 正候选还在 full pool，但不在 Top-5 |
| `ranking_loss_top5` | 1,954 | Top-5 有正候选，但 q Top-1 选错 |
| `already_correct` | 2,833 | q Top-1 严格口径正确 |
| `valid_empty` | 55 | 合法空候选样本 |
| **合计** | **7,675** | 互斥且覆盖全部样本 |

这个漏斗给出一个重要上限：只重排 frozen Top-5 最多能处理
`ranking_loss_top5`；它无法修复候选根本不存在、被 mask/NMS 删除或正候选
在 Top-5 之外的样本。

## 4. 数据 split 与 test 泄漏审计

来源：

`outputs/modular_reranking_v1/20260728T210046+0100/split_audit/split_audit.json`

SHA-256：

`a8a18ba4961edb04ebf83b6aa0fd4bcb02dfa23c622797f37e08252a881301f2`

| split | expressions | scene-frame | manifest SHA-256 |
|---|---:|---:|---|
| train | 26,295 | 1,104 | `a986bcce3e1961be816a295c3ae0942e64e61275524a85c0a8957563e7f920c1` |
| validation | 3,778 | 165 | `573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a` |
| test | 7,675 | 325 | `915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409` |

train–val、train–test、val–test 的以下交集全部为 0：

- `sample_id`
- `scene_id`（scene-frame ID）
- RGB SHA-256
- depth SHA-256
- RGB-D pair SHA-256

相同 sequence path 会跨 split 重现，因此准确表述是
**scene-frame/RGB-D image-disjoint**，不能表述为 unseen-sequence。

固定上游 provenance：

- HiFi checkpoint SHA-256：
  `436a54ecc159a36664f55f762463c54fc9b082f44205cee8020bed59fb5280d0`
- Dex-Net config SHA-256：
  `899e5295a83bed50f47dbc212028485d77637ed4acb56c075d132e43f88d0841`
- GQ-CNN model manifest SHA-256：
  `8201961abe3a09d90c6c66e582a3bfeb181d7095a2ebcc3a9d90e68fc12e8614`

## 5. Compact pipeline 一致性

### 5.1 冻结 HiFi 10-sample smoke

`outputs/modular_reranking_v1/20260728T210046+0100/hifi_test_smoke10/summary.json`

SHA-256：
`18f3da77f79a94b11096a4aaa039efc445f8956249c33c8792d04b2dee19bac9`

结果：

- 10/10 完成；
- `reference_probability_max_abs_difference = 0.0`；
- `reference_native_mask_mismatch_px = 0`；
- `gt_artifacts_exported = false`；
- MPS float32，elapsed 1.392 s。

### 5.2 20-sample 全链路 repeat

`outputs/modular_reranking_v1/20260728T210046+0100/determinism_val_repeat20_full_pipeline.json`

SHA-256：
`6cfda0ffb09fa482344e378c12f0bee232a1f7ea2f14ac2cd68efaf3419468bf`

结果：

- 20 个 probability arrays 完全一致；
- 20 个 native masks 完全一致，mismatch 0 pixels；
- 529 个候选集合和 pose 完全一致；
- 529 个 q-value arrays 完全一致；
- 最大 pose 差和最大 q 差均为 0。

### 5.3 100-sample shard/merge pilot

`outputs/modular_reranking_v1/20260728T210046+0100/shard_merge_val_pilot100_identity.json`

SHA-256：
`e7b4a14aae62e33f74284a683611849f3aa2713837f913cfd1d40d8f8f20066e`

四个 scene-hash shard 合并后，100 个样本、1,921 candidates 与单进程结果
逐元素一致；最大 pose 差为 0。

Dex-Net pilot 配置：

`outputs/modular_reranking_v1/20260728T210046+0100/dexnet_val_pilot100/run_config.json`

SHA-256：
`4b42430bfff5af55c0444b342a136387720218203eeb4bbc2ce3e0c650625f82`

该 pilot 产生 raw 17,376、mask-valid 17,300、NMS 1,921，0 execution
failures。GQ-CNN 统计：

`outputs/modular_reranking_v1/20260728T210046+0100/gqcnn_val_pilot100/run_statistics.json`

SHA-256：
`3eceff5112f857fde978c3fa3bfd812328a8cdb30a09b8e47abdc0bb763708ba`

1,921/1,921 q-values 有限，0 failed samples。

## 6. 特征和 GT 防泄漏

100-sample validation feature pilot：

`outputs/modular_reranking_v1/20260728T210046+0100/features_val_pilot100/dataset_manifest.json`

SHA-256：
`4ff28e483952245332a721884b292cc9c383facc5ca0604be2dabc5946eb2e2d`

已验证：

- 100 samples、1,921 candidate rows；
- 69/69 inference features 有限；
- `(sample_id, candidate_id)` 唯一；
- inference row 在 GT label join 前单独哈希；
- `labels_joined_post_extraction = true`；
- allowlist 与 forbidden GT columns 无交集；
- visible geometry 字段带有“可见表面 proxy、不是完整碰撞保证”的声明。

这证明特征提取器在 pilot 上遵守接口；full development/validation 数据质量审计
仍为 `PENDING`。

## 7. Local VLM 安全边界

审计产物：

`artifacts/experiment/modular_reranking_v1_20260728T210046+0100/local_ollama_audit_v2.json`

SHA-256：
`7fd3f799da95fc252f413081383f4795afe465e72db1c0b35fa446af12d85fc6`

已验证：

- Ollama 0.24.0；
- endpoint `http://127.0.0.1:11434`；
- listener 仅 `127.0.0.1:11434`；
- `disable_ollama_cloud=true`；
- 日志包含 `Ollama cloud disabled: true`；
- 没有 remote established connections；
- exact model `qwen3-vl:4b-instruct-q4_K_M`；
- manifest SHA-256
  `ee4b975b58c17ce268cd19d40db35d5edc64603035d2ffc1fee1968eb0947f7b`；
- model layer 3,295,612,928 bytes，Q4_K_M。

完整说明见 `docs/MODULAR_RERANKING_V1_VLM.md`。

## 8. 测试

2026-07-28 在当前工作树重新运行：

```text
/opt/anaconda3/bin/python -m pytest -q \
  tests/test_modular_reranking_v1.py \
  tests/test_modular_reranking_v1_split_audit.py \
  tests/test_modular_reranking_v1_features.py \
  tests/test_modular_reranking_v1_models.py \
  tests/test_modular_reranking_v1_vlm.py \
  tests/test_modular_reranking_v1_evaluation.py \
  tests/test_modular_reranking_v1_experiment_lock.py \
  tests/test_modular_reranking_v1_gallery.py \
  tests/test_gqcnn_ranking_evaluation.py
```

结果：**90 passed in 14.00 s**。这是上述命令在该次文档快照的真实终端
结果；不是对用户要求的“30 项测试”逐项宣称全部覆盖。

本次 focused run 没有单独写入持久化 pytest log；可持久校验的是上面的完整
命令和下列测试源码 SHA-256。正式交付仍应把 full test 命令与输出写入
`commands.log`，当前为 `PENDING`。

测试源码 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `tests/test_modular_reranking_v1.py` | `4cc5fa80f9c7bf4dda3802d81b36b9dbac1584ff1faddbc3eae75e6f752d6d69` |
| `tests/test_modular_reranking_v1_split_audit.py` | `7322d415500dd86bae9fd29c3c11eea9c7d1edc5b223d56b74a9b2781530f7cd` |
| `tests/test_modular_reranking_v1_features.py` | `12ef3d16dabb70fc965cab533de7c4ebc7a73b069e37a014f2c613ba9eb41650` |
| `tests/test_modular_reranking_v1_models.py` | `80e93dc190b1e353722482c2443d981a06b6777b754907c09ca5b28412596444` |
| `tests/test_modular_reranking_v1_vlm.py` | `e85bd2769a1392a7105d958db2cc94c3ca5b34346b1ead918860f62e2a4deeda` |
| `tests/test_modular_reranking_v1_evaluation.py` | `2ac01f4b72d1674eb4fb135212a5dbb73c1e32bc72ff2bf4cb10c585f7c916fa` |
| `tests/test_modular_reranking_v1_experiment_lock.py` | `4c2f52e2b8c4fa759d87d73a1edbec988ec426e32d95c80be288b7069e9d42a3` |
| `tests/test_modular_reranking_v1_gallery.py` | `4c77339bb9fed981a69ce7af88709c777e8f29646d5c44359b2bd45aa9b06086` |
| `tests/test_gqcnn_ranking_evaluation.py` | `de9587b62cb00cea7d4a5084ff47a5ec28ebbee51aee2b1f763149ea93a5be78` |

从实际收集到的测试名和断言，可以明确声称已覆盖：

- 严格 IoU/angle 边界和 180° periodic angle；
- 候选 identity 只允许 permutation，pose 修改会被拒绝；
- Funnel 类别顺序、互斥性和 valid-empty score marker；
- split audit 接受 disjoint assets，并拒绝重复 RGB bytes；
- soft-mask 双线性插值、图像边界、左右 jaw、width/normals；
- 可见点云变换和无 depth 时 collision proxy 仍为有限值；
- NPZ row 0 不能被当成 Top-1、train-only scaler、allowlist 和 GT 列拒绝；
- `(sample_id, candidate_id)` 标签 join；
- 规则确定性、geometry penalty、sample-balanced pairwise、all-negative、
  listwise、bounded residual、DeepSets permutation equivariance 和
  scene-grouped OOF；
- safe-switch 类的 OOF 训练、threshold 和 fallback；
- VLM GT-free visualization、loopback endpoint/payload、schema parser、
  invalid/missing/duplicate ID、non-finite score、timeout、abstain、cache
  tamper 和 q-only fallback；
- evaluator 的 denominator/outcome 复算、candidate pool/identity、Protocol B
  q baseline、exact McNemar、Holm、scene bootstrap、VLM runtime 和 bundle
  复算；
- experiment-lock exclusivity 和 identical-resume guard；
- gallery GT 隔离和 quota shortfall fail-closed；
- 旧 GQ-CNN evaluator 的 finite q、rank、multiple GT “match any one”、
  candidate ID/pose mismatch、separate immutable roots 和真实输出 regression。

### 8.1 明确的测试缺口

以下要求不能从上述 90-test 命令中证明，当前必须写为 `GAP/PENDING`：

| 要求 | 当前证据 | 状态 |
|---|---|---|
| 同一个 GT 同时满足 IoU+angle，且禁止跨 GT 拼接 | evaluator 实现和真实 baseline 使用 same-GT predicate，但没有专门构造“GT-A 只过 IoU、GT-B 只过 angle”的单元测试 | `GAP` |
| polygon clipping、`x > 480`、明确的 row/column 反例、empty GT | 当前 focused test 名和断言没有逐项构造这些边界 | `GAP` |
| depth hash leakage 专测 | split 实际产物显示 depth hash 交集为 0；单元测试只明确注入重复 RGB bytes | `GAP`（单测） |
| valid-empty 的完整 feature extraction | 有 valid-empty score marker 和 evaluator denominator 测试，但没有 end-to-end full feature row 测试 | `GAP` |
| `audit_local_ollama.py` 脚本级测试 | 有真实 local audit JSON 和 VLM backend unit tests，但没有脚本级 fixture test | `GAP` |
| representative VLM selector CLI | 实现存在；上述命令没有 selector CLI 测试 | `GAP` |
| VLM safe-switch selection CLI | safe-switch 类有 synthetic unit test；选择脚本本身和 full validation threshold 尚未验证 | `PENDING` |
| full development/validation 和 formal-test 脚本 | 尚未完成正式运行和最终机器产物 | `PENDING` |

这些缺口不否定已完成的 baseline/split/pilot 证据，但在补测试或正式产物出现前
不能写成“30 项全部测试通过”。

## 9. 当前可以下的结论

1. 冻结 q-only baseline 可复现，且历史/严格阈值差异已定位到 evaluator
   边界，不是未知的坐标或排序错误。
2. test split、validation 和 train 在 scene-frame/RGB-D 层面隔离。
3. compact HiFi→Dex-Net→GQ-CNN pipeline 在已验证的 10/20/100-sample
   范围内复现且确定。
4. 候选身份和 GT 隔离有代码检查与测试保护。
5. 当前 baseline failure funnel 可以作为真实模型/候选池限制分析的起点。

## 10. 尚不能下的结论

- `PENDING`：哪一个 learned method 在 full validation 最好；
- `PENDING`：safe switch 是否满足 harmful-rate 上限；
- `PENDING`：VLM 是否优于 q-only 或 learned ranker；
- `PENDING`：正式 test 的任何新方法数值；
- `PENDING`：McNemar、Holm 和 10,000-draw scene bootstrap；
- `PENDING`：锁后 formal-test invocation 是否恰好为 1；
- `PENDING`：完整 recovered/harmful/VLM gallery quota。

因此，不应把 pilot 上的 regularized linear 或单次 VLM 输出称为最终 winner。

## 11. 风险与解释边界

- J@1 衡量的是与 OCID-VLG **二维抓取 annotation 的一致性**，不是物理抓取
  成功率；
- 单视角 depth 看不到物体背面和遮挡后的障碍；
- visible-surface collision proxy 不证明 collision-free；
- GQ-CNN-2.1 可能存在 domain shift；
- VLM 的视觉空间推理和结构化输出可能不稳定；
- 冻结 test 的历史 aggregate 和 candidate labels 已在本地暴露，所以最终
  报告必须如实说明它不是完全盲的 pristine test；但方法选择仍必须只看
  validation。
