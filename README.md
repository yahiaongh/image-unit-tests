# Image Unit Tests

**Self-verifying image generation agent using Muse Spark, Muse Image, and SAM 3.1.**

Image Unit Tests turns an image brief into explicit requirements, generates an image, and checks whether the result satisfies those requirements. When it finds a mismatch, it can give the image model a targeted correction and verify the result again.

## The Problem

Image generators can miss hard constraints: the exact number of objects, where an object appears, or whether something is absent. People are left manually counting objects, spotting mistakes, and re-prompting until the image looks right.

## The Solution

A closed-loop agent makes image generation measurable:

- **Muse Spark** turns the brief into structured constraints.
- **Muse Image** generates or edits the image.
- **SAM 3.1** segments requested objects.
- A **deterministic Python verifier** checks counts, presence, absence, and position.
- If a check fails, the agent creates a specific correction—for example, “The image contains 9 green apples, which is 1 too many. You must remove exactly 1 green apple.”—and sends it with the current image for editing.

The LLM proposes; deterministic Python decides whether the image passes.

## Benchmark: the loop beats blind retries

We evaluated 30 briefs per mode with live models. Each mode completed all 30 briefs without recorded API errors.

| Mode | Pass Rate | Avg Rounds (iterative modes) | Avg Cost per Image* |
|---|---:|---:|---:|
| One-shot | 70% (21/30) | — | $0.0173 |
| Pass@4 | 77% (23/30) | — | $0.0172 |
| Random Retry | 77% (23/30) | 1.97 | $0.0171 |
| **Loop** | **83% (25/30)** | **1.70** | **$0.0168** |

The closed-loop system improved pass rate by 6 percentage points over pass@4 and by 6 points over blind random retries. It averaged 1.70 image rounds per brief, compared with four independent generations for pass@4—less than half as many generation-and-verification rounds.

*Costs are configured Image/SAM estimates, not provider billing totals. Spark request costs are excluded. Pass@4 generates four images per brief; its pass rate is the fraction of briefs with at least one passing image.*

## Architecture

```text
                         +--------------------+
                         | User brief         |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | Muse Spark         |
                         | Constraint proposal|
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | Muse Image         |
                         | Generate image     |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | SAM 3.1            |
                         | Segment objects    |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | Python verifier    |
                         | Deterministic check|
                         +---------+----------+
                                   |
                      +------------+------------+
                      |                         |
                    PASS                      FIX
                      |                         |
                      v                         v
                Return result      Numeric instruction + image
                                                |
                                                v
                                        Muse Image edit
                                                |
                                                +----> SAM 3.1
```

## Run the demo

### Install

```bash
pip install -r requirements.txt
```

### Configure live API access (optional)

Copy the example environment file and add your Meta and Baseten credentials:

```bash
cp .env.example .env
```

Set `META_API_KEY`, `BASETEN_API_KEY`, and the Baseten configuration values required by your `config.yaml`. Keep `.env` private; it is git-ignored.

### Launch the UI

```bash
python -m src.ui.app
```

Select **Demo replay** to view the pre-captured **Apples** scenario offline. The saved Apples, Spatial, and Absence scenarios are included in `runs/demos/` and do not require API credentials to replay.

## Evaluation

The live evaluation harness supports one-shot, blind random retries, pass@4, and feedback-driven loop modes:

```bash
python -m src.eval.run_eval --mode one-shot --max-cost 4.00 --live
python -m src.eval.run_eval --mode random_retry --max-cost 4.00 --live
python -m src.eval.run_eval --mode pass@4 --max-cost 4.00 --live
python -m src.eval.run_eval --mode loop --max-cost 4.00 --live
```

The `--max-cost` guard estimates configured Image/SAM costs and excludes Spark costs. To run without network access or API credentials, omit `--live` to use the mock clients.

Run the test suite with:

```bash
python -m pytest -q
```
