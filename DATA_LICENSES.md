# Data and model licence boundaries for the GraspNet 6-DoF route

Last verified: 2026-08-15. This file is an engineering record, not legal advice.

## GraspNet-1Billion

- Official source: <https://graspnet.net/datasets.html>
- The official dataset page describes the dataset as non-commercial and labels
  its terms “Creative Commons Attribution 4.0 Non Commercial License
  (BY-NC-SA)”. That wording mixes different Creative Commons names, so a user
  must read and accept the current official page before downloading or using
  the data.
- Dataset archives are external inputs. They are stored under
  `downloads/graspnet/` and extracted under `data_external/graspnet/`; both are
  ignored by Git.
- This repository does not redistribute RGB-D scenes, annotations, collision
  labels, object meshes, or generated Dex-Net models.
- `configs/graspnet6d/graspnet_object_catalog_v1.json` is dataset-derived
  catalogue metadata and remains a local-only input. It is intentionally not
  redistributed under the repository's root MIT licence; recreate or retain it
  locally only after accepting the current official GraspNet terms.

## graspnetAPI and the evaluator

- Official source: <https://github.com/graspnet/graspnetAPI>
- The repository root at the pinned upstream revision declares MIT. The
  evaluator bundles/adapts Dex-Net components with research/non-profit notices;
  those narrower notices must be reviewed before redistribution or commercial
  use.
- `src/graspnet6d/evaluator.py` is a thin adapter. It does not modify the
  upstream tree and must not be described as a new or relaxed evaluator.

## VGN

- Official source: <https://github.com/ethz-asl/vgn>
- The VGN source tree at the pinned `corl2020` revision is BSD-3-Clause.
- The upstream pretrained checkpoint is an external model asset. The official
  repository does not state a separate model-weight licence, so this project
  records the checkpoint hash but does not infer that the code licence grants
  additional rights for the weights.
- VGN remains frozen. This route adds an adapter and does not edit the vendored
  upstream snapshot.

## HiFi-CS checkpoint

- The local checkpoint is an output of the existing reproduction route. Its
  use remains subject to the licences and usage terms of its training data,
  CLIP backbone, and repository dependencies.
- It is not redistributed by this repository (`*.pth` is ignored).

## LightGBM

- Official source: <https://github.com/lightgbm-org/LightGBM>
- LightGBM source is MIT. It is installed into an isolated local environment;
  no third-party source is copied into this repository.

## Derived artifacts

Language expressions, candidate caches, labels, feature tables, figures, and
reports derived from GraspNet data remain under `artifacts/graspnet6d/` and are
ignored by Git. Their distribution may still be constrained by the source
dataset terms.
