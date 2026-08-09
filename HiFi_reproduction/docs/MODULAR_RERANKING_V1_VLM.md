# Modular Candidate Re-ranking V1：本地 VLM 说明

> 文档状态：local-only 部署审计和 1/10-sample engineering smoke 已完成。  
> 100-sample representative pilot、独立 20-sample repeat、full validation、
> validation 选择、实验锁和 formal test 仍为 `PENDING`。

## 1. VLM 在实验中是什么

VLM 是一个 **training-free local comparator**。它只对冻结的 GQ-CNN Top-5
重新排序，不生成新抓取，也不修改抓取几何。

它每个样本只调用一次，收到：

- 自然语言指令；
- `full_scene_overlay.png`；
- `candidate_contact_sheet.png`；
- frozen Top-5 candidate IDs；
- visual+metadata 版本才会额外收到允许的推理摘要。

它看不到 GT mask、GT grasp、candidate correctness、GT IoU/angle 或正确
candidate ID。

> **像给 12 岁孩子解释**
>
> 五个候选像五张已经写好编号的答题卡。VLM 可以看题目和图片，重新排
> 1–5 名，但不能画第六张卡，也不能擦掉或改动任何一张卡。看不清时，它应该
> 说“不确定”，系统就用原来的第一名。

## 2. 为什么选这个本地模型

当前 pin 的模型是：

- backend：Ollama；
- version：0.24.0；
- exact model：`qwen3-vl:4b-instruct-q4_K_M`；
- parameter scale：4B；
- quantization：Q4_K_M；
- local model layer：3,295,612,928 bytes；
- Ollama manifest SHA-256：
  `ee4b975b58c17ce268cd19d40db35d5edc64603035d2ffc1fee1968eb0947f7b`；
- model layer SHA-256：
  `16b83be682148a4d8201dbf720ea7eace5de98b69f63f05e0c908b4d7977ecb5`；
- license blob SHA-256：
  `7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2`。

官方 Ollama tag 页面记录该 tag 支持 text/image 输入、4.44B、Q4_K_M、
约 3.3 GB，digest 前缀 `ee4b975b58c1`。本机完整 manifest hash 与之对应。

官方资料：

- [Ollama Qwen3-VL exact tag](https://ollama.com/library/qwen3-vl%3A4b-instruct-q4_K_M)
- [Ollama Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags)
- [Qwen3-VL official repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL Apache-2.0 license](https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE)

8B 只能在 validation pilot 证明内存稳定、延迟可接受且有明确增益后再进入
锁定比较。当前没有运行 8B，也没有根据 test 选择模型。

## 3. Local-only 证据

机器可读审计：

`artifacts/experiment/modular_reranking_v1_20260728T210046+0100/local_ollama_audit_v2.json`

SHA-256：

`7fd3f799da95fc252f413081383f4795afe465e72db1c0b35fa446af12d85fc6`

审计记录：

- server config：`{"disable_ollama_cloud": true}`；
- config SHA-256：
  `30c5a0e23ac2015aa3fb9a17391e1ab72fa5668ff9bd37cd3caccc864c8e21ec`；
- endpoint：`http://127.0.0.1:11434`；
- listener：只绑定 `127.0.0.1:11434`；
- 日志：
  `Ollama cloud disabled: true`；
- 日志：
  `Listening on 127.0.0.1:11434 (version 0.24.0)`；
- remote established connections：空；
- API transport：`no_proxy_no_redirect_loopback_http`；
- remote API used：false。

本机原来不存在 `~/.ollama/server.json`，因此没有旧文件可备份；新配置的
存在和 SHA 已记录。

Ollama 官方 FAQ 说明可以通过 `disable_ollama_cloud=true` 或
`OLLAMA_NO_CLOUD=1` 关闭 cloud，并在日志中看到相同确认文字：
[Ollama local-only FAQ](https://docs.ollama.com/faq)。

## 4. 两张 GT-free 输入图

### 4.1 Full scene overlay

包含：

- 原始 RGB；
- predicted HiFi mask 半透明 overlay 和 contour；
- GQ-CNN Top-5 rectangles；
- candidate ID；
- 原 q Top-1 的中性标记。

不包含 GT、正确/错误颜色或 GT target mask。

### 4.2 Candidate contact sheet

每个候选使用相同布局显示：

- RGB crop；
- metric depth crop；
- depth-edge crop；
- predicted mask crop；
- rectangle、jaw/contact regions 和 candidate ID。

深度色标在同一样本内一致，避免仅因每个 crop 自动拉伸就产生误导。输入图与
候选 ID 一一对应。

### 4.3 Visual 与 visual+metadata

`local_vlm_visual` 只给 instruction、两张图和 ID。

`local_vlm_visual_metadata` 额外给：

- q percentile；
- soft-mask support；
- width compatibility；
- jaw depth difference；
- visible clearance proxy；
- candidate cluster size。

这些摘要来自 inference allowlist，不包含 GT。

## 5. Prompt 的行为边界

system prompt 要求模型：

1. 只能排序给定 candidate IDs；
2. 先根据语言找目标，再检查 jaw contact、width fit、depth continuity、
   depth boundary 和 visible clearance；
3. 不确定时保留原 GQ-CNN Top-1 或 abstain；
4. 不能宣称物理抓取成功；
5. 只返回 JSON schema 对象。

推理固定：

- `temperature = 0.0`；
- `seed = 20260728`；
- `stream = false`；
- `think = false`；
- `max_output_tokens = 768`；
- 每个样本独立 request，不保留聊天历史；
- one request at a time。

Ollama 官方 structured-output 文档说明 `format` 可接收 JSON schema：
[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)。

## 6. JSON schema 和 fail-closed 校验

响应必须包含：

- `selected_candidate_id`；
- 唯一且完整的 `ranking`；
- 每个 candidate 的有限 score 和 reason codes；
- `[0,1]` 内的 confidence；
- `abstain`；
- `switch_from_original_top1`。

以下任一情况会回退到 q-only：

- parser/schema failure；
- invalid or duplicate candidate ID；
- 缺少候选；
- 非有限 score；
- timeout；
- `abstain=true`。

fallback reason、raw response、parsed response、request hash、latency、
token counts 和 Ollama timing 均保存。

> **为什么要“失败就回原答案”？**
>
> VLM 像一个可能看错图的同学。如果它写了不存在的候选编号、漏答、格式坏掉
> 或主动说不确定，系统不会猜它“可能想说什么”，而是稳妥地使用原 GQ-CNN
> 第一名。

## 7. Cache 与可重复性

cache key 对以下内容做 SHA-256：

- model digest 和 backend version；
- system/user prompt；
- 输入图；
- candidate metadata；
- generation options。

相同 request hash 会读取已保存 raw/parsed response，不再次调用模型。
cache hit 和 newly processed 分开计数。

单样本 cache replay：

`outputs/modular_reranking_v1/20260728T210046+0100/vlm_val_smoke1_cache_replay/summary.json`

SHA-256：
`8de1c23b2bc72476af2dfeca7461246c1d0717c6c63b2a3e4d325d74133ab43c`

结果：

- `cache_hits=1`；
- `newly_processed=0`；
- wall time 0.00843 s；
- 没有新模型调用。

这验证的是 cache replay，不等同于两个独立 fresh inference 完全一致。
20-sample 两次独立 fresh inference 的 deterministic agreement 仍为
`PENDING`。

## 8. 已完成的真实 smoke

### 8.1 单样本

`outputs/modular_reranking_v1/20260728T210046+0100/vlm_val_smoke1_visual/summary.json`

- 1/1 structured response 通过；
- 0 fallback；
- wall time 11.356 s；
- prompt tokens 1,374；
- output tokens 351；
- Ollama response 含 load/eval/total duration；
- 使用两个真实输入图和 pinned local model。

这一次输出只证明调用链和 schema 有效，不证明候选选择正确率。

### 8.2 10 样本

`outputs/modular_reranking_v1/20260728T210046+0100/vlm_val_smoke10_visual/summary.json`

SHA-256：
`7e599cd90186ad59c55e46ad975b585662db30a1c9c31fa35dcb09308a7387ce`

结果：

- 10 samples；
- 10/10 没有 fallback；
- 1 cache hit，9 newly processed；
- total wall time 151.051 s；
- exact model/digest、temperature、seed、stream、think 均写入 summary。

这是 engineering smoke，不是 representative 100-sample pilot，也不能用于
宣称 VLM 相对 q-only 有提升。

## 9. 尚未完成的正式 VLM 表

| 项目 | 状态 |
|---|---|
| representative 100-sample coverage audit | `PENDING` |
| independent fresh 20-sample repeat | `PENDING` |
| deterministic agreement rate | `PENDING` |
| visual-only full validation | `PENDING` |
| visual+metadata full validation | `PENDING` |
| confidence/safe-switch threshold | `PENDING` |
| valid JSON / invalid ID / abstain / fallback / timeout rates | `PENDING` |
| mean/p50/p95 latency | `PENDING` |
| total wall time / samples per hour | `PENDING` |
| memory peak / swap audit | `PENDING` |
| validation-selected VLM comparator | `PENDING` |
| locked formal-test VLM result | `PENDING` |

## 10. 解释边界

即使 VLM 在 2D 指标上选对候选，也只能说它与 OCID-VLG 的 2D rectangle
annotation 一致。VLM 看到的是 RGB、预测 mask 和可见单视角深度可视化：

- 看不到完整 3D 物体和遮挡后空间；
- 不能证明 force closure；
- 不能证明 collision-free；
- 不能证明机器人 reachability；
- 不能证明实际 lift success。

Safe Switch 中冻结的 `collision_proxy_total <= 0.5` 也只是一项
**visible-surface collision/clearance proxy**：它只检查单视角深度图里已经
看见的表面。它看不到遮挡后的物体，也不包含完整机器人、桌面、工作空间、
运动轨迹或自碰撞模型，所以绝不能解释成完整的无碰撞、可达性、
force-closure 或机器人安全保证。

因此 prompt 和报告都禁止使用“VLM 证明物理抓取成功”之类表述。

## 11. 外部复用决定

官方网页只用于验证 tag、digest、最低 Ollama 兼容要求、image input、local-only
配置、structured output 和 Apache-2.0 license。没有从外部仓库复制业务逻辑或
实验代码；本地后端、schema validator、cache、fallback 和可视化均按本仓库
冻结 candidate contract 实现并测试。
