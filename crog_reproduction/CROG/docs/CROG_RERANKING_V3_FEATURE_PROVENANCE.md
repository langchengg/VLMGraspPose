# CROG Re-ranking V3 Feature Provenance

提取期机器记录：`failure_analysis/reranking_outputs/v3_fullchain_20260801T114301+0100/provenance/V3_FEATURE_PROVENANCE.json`；最终 inclusion decision 由冻结的 `formal_inputs/experiment_descriptor.json` 与 `diagnostics/feature_ablations/feature_ablation.json` 给出。所有组均被提取/验证，但 primary 只采用在 `v3_select` 选择的 G0–G6。

| Feature group | Source tensor | Candidate-specific | V2 primary | V3 extracted | V3 primary | Decision reason |
|---|---|:---:|:---:|:---:|:---:|---|
| G0 | frozen q/candidate geometry | yes | yes | full | included | frozen candidate identity and bounded-residual base |
| G1 | raw + activated Q `[B,1,104,104]` | yes after ROI | no raw map | full | included | selected head evidence |
| G2 | raw + sigmoid M `[B,1,104,104]` | yes after ROI | only restored/crop derivative | full | included | removing mask reduced select J@1 by 0.174 pp |
| G3 | sin/cos heads `[B,1,104,104]` | yes after ROI | only restored/crop derivative | full | included | removing angle confidence reduced select J@1 by 0.213 pp |
| G4 | raw + activated W `[B,1,104,104]` | yes after ROI | only restored/crop derivative | full | included | removing width consistency reduced select J@1 by 0.270 pp |
| G5 | M/Q/angle/W joint evidence | yes | no explicit joint vector | full | included | largest leave-one-out loss: 0.290 pp |
| G6 | C3/C4/C5/FPN/decoder/projector ROI | yes after ROI | post only | full | included | selected all-latent adapter; layer ablation reported separately |
| G7 | sentence EOT | query-global conditioning | indirect only | full | conditioning consumed, but not counted as a candidate-token feature group | sentence mode was best select configuration; the selected feature-group set remains exactly G0–G6 |
| G7 | token/cross-attention/dynamic interaction | yes after ROI interaction | no | full | excluded | token mode was 0.580 pp below sentence mode and had 38 vs 16 harmful switches |
| G8 | aligned RGB + predicted maps | yes | yes, limited 14ch crop | full | excluded | aligned crop configuration underperformed selected primary |
| G9 | aligned depth + geometry | yes | partial | full | auxiliary RGB-D only | RGB-D was 0.155 pp below native on select; kept separate for modality audit |
| G10 | pairwise/set relations | yes | six relations | full | excluded | added complexity without surpassing the simpler primary |
| G11 | V2 OOF base/SetRank/gate | yes | n/a | 53,431 OOF rows verified | anchor/gate only | scorer prior was not selected; V2 remains conservative deployment anchor |

“Full-chain”在本项目中表示所有预声明、部署可得的 CROG-native signal 都被识别、提取并受控验证，不表示信息论意义上的绝对完全利用。depth 是 V3 的额外传感增强，不是 CROG-native input。
