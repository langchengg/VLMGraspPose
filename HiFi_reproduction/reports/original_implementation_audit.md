# Original HiFi-CS implementation audit

Label for this work: **Official HiFi-CS repository reproduction on macOS/MPS**.

It is not an exact paper reproduction because the released repository and paper disagree in scientifically material ways and the paper checkpoint/manifests are not published.

## Official repository commands

- Prepare OCID-VLG: `python datasets/prepare_ocidvlg.py`
- Train OCID-VLG: `python training.py config.yaml 1`
- Evaluate OCID-VLG: `python score.py config.yaml 1 1`

## Repository configuration

| Item | Released repository |
|---|---|
| Model | `models.hifics.CLIPDensePredT` |
| CLIP image/text encoder | OpenAI CLIP `ViT-B/16` |
| Visual blocks | `[1, 3, 5, 7, 9]` plus block 0 retained as a returned feature |
| Decoder | Five `TransformerEncoderLayer` blocks, linear reductions, FiLM, one transposed convolution |
| Decoder width | 64 |
| Input | 352 x 352 RGB |
| Loss | binary cross entropy with logits |
| Optimizer | AdamW |
| Scheduler | cosine annealing, `T_max=20000`, `eta_min=0.0001` |
| Learning rate | 0.001 |
| Batch size | 16 |
| Optimizer updates | 20,000 |
| Periodic saves | requested at 5k/10k/15k/20k; released loop exits before the 20k periodic save |
| Final save | trainable-only `weights.pth` |
| Evaluation threshold | OCID config says 0.85 |
| Dataset protocol | OCID-VLG `unique` train and test manifests; validation is ignored |
| Resume | not supported in released code |

## Paper configuration

The paper states a frozen CLIP image/text encoder, selected blocks K={1,3,5,7,9}, hierarchical FiLM before every decoder block, roughly 6M trainable parameters, pixelwise BCE, Adam, cosine scheduling, and a 70/30 OCID-VLG split. The exact checkpoint, exact 70/30 manifests, threshold implementation, batch size, learning rate, and full training recipe corresponding to Table 2 are not published.

## Verified discrepancies and defects

- The paper describes hierarchical FiLM, but the repository default has `extended_film: False`, so it conditions only decoder block 0.
- The paper says Adam; the repository uses AdamW.
- The paper says 70/30; the repository preparation uses the dataset's published `unique` 70/10/20 train/val/test manifests and omits validation.
- The paper describes a binary softmax output; the repository uses one-channel BCE-with-logits.
- `score.py` computes sigmoid probabilities but ignores them and thresholds raw logits.
- The shared loader shuffles every split, including evaluation.
- The released training loop has no seed, full-state checkpoint, or resume path.
- `general_utils.load_model` defaults to CUDA mapping.
- `training.py` selects CUDA or CPU only and uses CUDA-specific AMP.

## Parameter freezing

`CLIPDenseBase` explicitly sets all `clip_model` parameters to `requires_grad=False`. FiLM projections, decoder reductions, transformer blocks, and transposed convolution remain trainable. This must still be verified by an instantiated model before full training.

## Reference result only

Paper Table 2 reports IoU 88.26, P@50 92.68, P@60 92.13, P@70 91.53, P@80 89.69, and P@90 83.21. These are comparison targets, not local results.

