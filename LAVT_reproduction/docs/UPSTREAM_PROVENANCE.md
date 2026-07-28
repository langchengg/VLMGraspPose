# Upstream provenance

- Upstream URL: https://github.com/yz93/LAVT-RIS.git
- Upstream branch: `main`
- Upstream commit: `1da0af9f21b637c0cae9ea1363d2dd9b40e19628`
- Retrieved: 2026-07-25 (Europe/London)
- Upstream license: GNU General Public License v3.0 (`LICENSE` retained unchanged)

## Local modification scope

The local work adapts the official LAVT-RIS implementation to the OCID-VLG
visual-grounding task. It adds an OCID-VLG RGB/text-to-mask dataset adapter,
paired RGB/mask transforms, manifest and data audits, current CUDA/MPS/CPU
device handling, Dice and device-safe weighted cross-entropy losses, resumable
training, original-resolution evaluation, prediction export, tests, and
reproduction documentation.

The LAVT backbone, PWAM language-aware fusion, BERT encoder, and mask predictor
remain the official architecture. No grasp, depth, SAM, re-ranking, or
ground-truth-guided inference component is added.

The OCID-VLG API repository has no declared license. Its source is not copied
into this repository; the adapter dynamically imports an already installed
local API and calls its dataset implementation.

