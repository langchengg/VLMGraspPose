# Mac smoke-test result

The gate passed on MPS in float32 with CPU fallback disabled.

| Check | Result |
|---|---|
| repository/PyTorch imports | pass |
| MPS availability | pass |
| OpenAI CLIP ViT-B/16 load | pass |
| OCID-VLG sample and DataLoader batch | pass |
| language tokenization | pass |
| model forward, shape `(1,1,352,352)` | pass |
| finite BCE-with-logits loss | pass (`0.683642`) |
| backward and AdamW optimizer step | pass |
| full-state checkpoint save/reload | pass |
| short evaluation and IoU computation | pass |
| accidental CUDA path | none observed; CUDA unavailable |

Measured one-sample forward time was 0.602 seconds, including first-run effects. The untrained one-step IoU (`0.00306`) is a smoke value and must not be reported as a model result.

The instantiated repository configuration has 151,732,162 total parameters and 2,111,425 trainable parameters. All CLIP image/text parameters are frozen. The local trainable count is materially below the paper's approximate 6M claim and is another paper/repository discrepancy.
