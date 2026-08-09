# CROG Re-ranking V3 Implementation

状态：阶段 1–6 已完成；唯一正式 test pass 已在冻结 manifest 下运行。所有结果仍以 `failure_analysis/reranking_outputs/v3_fullchain_20260801T114301+0100/` 内的不可变 artifact 为准。

## Artifact DAG

`Frozen CROG inputs → forward audit → candidate identity → raw input/affine → output-map evidence → multiscale latent ROI → token-candidate inputs → aligned RGB/RGB-D crop → provenance-valid V2 OOF prior → FCER candidate embeddings → grouped OOF base predictions → V2-anchored gate → ensemble/perturbation uncertainty → ranking of original candidate IDs → independent corrected+legacy evaluation`

每个完成 artifact 记录 schema/version/status、parent/input hashes、candidate/checkpoint/split/evaluator/code/config/schema hashes、seed/device/dtype、行数/唯一 ID 数、missing/fallback、时间和 content SHA。写入限定在新 `v3_fullchain_<run_id>` 根，原子 rename；manifest complete 后复用必须 fingerprint/hash 一致。

## 坐标约定

所有候选几何的源坐标为原始图像 `[x,y]`。映射顺序是：

1. 由 dataset 的真实 forward affine 把原图 homogeneous point 映射到 416×416 input；
2. 从 input pixel centre 映射到目标 feature-map pixel centre；
3. 以候选轴为局部 x-axis 构建 normalized sampling grid；
4. `grid_sample` 明确使用 `(x,y)`、`align_corners=False`、bilinear/zeros；
5. axial angle 以 modulo π 处理。

inverse affine 仅用于把 native/restored map 返回原图；不能把 forward/inverse 混用。MPS 不支持或数值 probe 失败时，单个 ROI 操作回退 CPU，不更改 CROG forward。

## Feature groups

- G0：冻结候选、q 和 list-level scalar；
- G1–G5：Q、predicted mask、angle、width 与五头 cross-consistency；
- G6：C3/C4/C5/FPN/pre/layer1/layer2/layer3/post/projector branch 的候选对齐池化；
- G7：candidate ROI query 对 Ft 的 masked cross-attention、sentence compatibility 与 dynamic-kernel/ROI interaction；
- G8：gripper-axis aligned RGB + predicted-map crop；
- G9：额外 RGB-D geometry，始终与 native 分开报告；
- G10：无位置编码的 permutation-equivariant set context；
- G11：只有 producing fold 与 group provenance 合法的 V2 OOF evidence。

所有组都必须完成 extraction/validation；最终状态只能是 included、redundant、harmful、unavailable 或 provenance invalid。

## FCER

候选架构 FCER-Native 使用 G0–G8、G10 和合法 G11；FCER-RGBD 额外使用 G9。统一 candidate encoder 包含 scalar/output-map/latent/token/crop/depth adapters，hidden dim ≤256，两层、四 heads、float32 set encoder，无位置编码。`v3_select` 最终选择的 primary 是更简单的 `sentence_only`：G0–G6、hidden dim 128、三种子 ensemble、`alpha=0.5`，不使用 crop/depth/set/V2-prior scorer；V2 prior 只作为冻结 anchor 和 gate evidence。

基础分数为：

`score_i = logit(clip(q_i, eps, 1-eps)) + alpha * tanh(r_i)`

`alpha=0` 或 residual 全 0 必须 bit-exact 回到 q ordering。tie-break 固定为 score、原 q-rank、candidate ID。

训练损失由 listwise、多标签 absolute correctness、positive-negative pairwise、query any-positive 与 residual regularisation 组成；Top-5 全错样本保留，listwise positive 项跳过但三个 absolute/any/gate 监督仍有效。

## Token-candidate interaction

query-global sentence vector 本身不能区分同 query 的候选。每个候选 ROI 生成 query vector，对非 SOT/EOT/pad token 做 cross-attention，并产生 candidate-conditioned text vector、attention entropy 与 global compatibility。完全相同的 ROI 必须输出相同 interaction；不同 ROI 应可不同。任何文本词汇规则只从 development 或预声明常识表构建，未知词稳定 fallback；symbolic program 不进入模型。

## OOF 与 V2-anchored gate

固定三 folds，group unit 为 capture sequence。每条 OOF 记录必须保存 producing fold、checkpoint SHA、fit groups hash 与 held-out group，并验证交集为 0；normalisation、adapter、feature selection 和 calibration 都在 fold 内拟合。

primary gate 默认 baseline 为 V2 selected candidate。challenger 的 `pR,pH,pN` 构成 `gain=pR-lambda*pH-kappa_u*uncertainty`；只有 gain 超过 tau、三 seed consensus 达标、coverage 合法且 identity 通过才覆盖 V2，否则精确保持 V2 selection。q-anchored 仅作为消融。

## Uncertainty 和 fallback

预声明 seeds 为 20260801、20260802、20260803。perturbation 为 center ±2/±4 px、angle ±5/±10°、width ±5/±10%；只重采 evidence，不添加或修改最终候选。保存 mean/std/min/max/valid fraction/rank consistency/top-1 vote/ensemble disagreement/probability variance。

fallback 层级：RGBD 缺失→Native；Native/coverage/NaN/extraction 失败→V2 exact selection；V2 identity 或 lock 失败→阻断，不输出伪结果。

## Test access 与 schema separation

`TestAccessGuard` 对 development/calibration/select 拒绝 formal-test、test-label 和 test-ranking 路径，并写 append-only access journal。inference 与 label artifact 使用不同 loader；训练 loop 才显式 join。inference field/lineage 通过严格 allowlist，禁词包括 GT、answer、objID、target index、label/success/correctness、IoU、angle error、Oracle 和 evaluation result。

正式 test 必须先校验 final manifest 及 `.sha256` sidecar，并原子 claim 一次；dry-run 不能生成任何有效 lock 或 claim。

## CLI 与测试状态

统一 CLI 已实现并实际运行任务要求的 25 个 command，包括 audit、extraction、OOF/final training、calibration、selection、diagnostics、preliminary/final lock、formal inference、independent evaluation、gallery 和 report 生命周期。长任务使用原子完成文件与 `--resume`；formal lifecycle 用一次性 claim 防止第二次有效运行。

V3 focused suite 最终为 225 passed、2 warnings；warning 仅来自 PyTorch `TransformerEncoder` 的 nested-tensor 优化提示。新增测试覆盖 identity/evaluator/affine/ROI/hook/token/equivariance/OOF/gate/fallback/lock/idempotency、float16 audit、subgroup/report evidence binding 等协议门槛。根目录旧测试 `test_diff_refer_types.py` 的 `engine.engine` collection 错误在 V3 前已存在，未通过放宽测试规避。
