# skill-dl-transformer-multiasset

Transformer-based multi-asset commodity-futures factor predicting next-5-day cross-sectional returns via PatchTST or iTransformer.

## Overview

This repository provides quantitative research tooling for Transformer-based multi-asset modeling of commodity futures.

**Core Functionality**:
- Joint modeling of multiple futures contracts using PatchTST / iTransformer
- Predict next-5-day cross-sectional returns
- Generate factor values for downstream strategies

## Directory Structure

- `dl-transformer-multiasset/` — Research and training code
- `dl-transformer-multiasset-production/` — Read-only production factor queries

## Usage

### Research and Training
Follow the instructions in `dl-transformer-multiasset/SKILL.md`.

### Production Factor Queries
Follow the instructions in `dl-transformer-multiasset-production/SKILL.md` (read-only).

## Dependencies

- Python 3.8+
- panda-data SDK
- PyTorch
- See subdirectories for additional dependencies

## Configuration Constants

- `FACTOR_ID = "DLTX"`
- `FACTOR_NAME = "Transformer多资产联合建模"`
- `DATA_VERSION = "real-v1"`

## License

GPL-3.0-only

## Boundaries

This repository contains research and engineering materials. **It does not constitute investment advice, does not promise returns, and does not represent official endorsement by QuantSkills / Panda data / Codex / Claude Code / Cursor / Hermes / OpenClaw.** Do not record or commit Panda data credentials.
