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

## Results

**We tested whether a segmentation-based verifier can close the loop on image generation. Numeric feedback steered the editor (human-verified error reduction +3.79 vs +0.74 for generic), but SAM couldn't certify the result (8 of 11 passes were wrong).**

On the human-verified N=19 overcount cohort, single-step edits had these
human-counted mean absolute error reductions:

| Condition | Mean Error Reduction (Human-Verified) |
| :--- | ---: |
| Numeric Feedback | +3.79 |
| Generic Feedback | +0.74 |
| Marked Image | +0.11 |
| Control | -0.37 |

Error reduction is the starting human-counted absolute error minus the edited
human-counted absolute error; positive values mean lower error after editing.
Numeric feedback's mean reduction was about **5.1x** Generic feedback's.

Among 11 edited images SAM passed, a blinded human count disagreed on 8
(72.7%, 95% Wilson CI 43–90%, n=11). The experiment records SAM counts and
human counts, but does not preserve the verifier's `UNCERTAIN` status for
these edited images. Therefore the UNCERTAIN catch rate cannot be calculated
from the saved evidence; we do not claim a catch count.

The SAM false-passes were off by exactly ±1 only 2 of 8 times. The signed
human-minus-expected deviations were +1 (2), -2 (2), +2 (2), +7 (1), and +12
(1). This sample does not support a claim that SAM is accurate to roughly ±1;
most false-passes were off by more than one object.

The image generator produced human-reported artifacts in 7/21 starting images
(33.3%); edited outcomes had artifacts in 25/84 images (29.8%). This is a
descriptive comparison, not a significance test. Only 1/21 Numeric outcomes
(4.8%) had both the expected human count and no reported artifact.

**The Yield Funnel:** 105 images generated → 84 edited → 11 SAM-passed → 3
human-correct → 1 artifact-free.

This is a hard-count pilot with n=19 hard overcounts and n=11 SAM passes.
Multi-round evaluation crashed and did not complete. Undercounts (n=2) were
excluded. Human labels were provided by a single human counter plus one
recounter.

**The Goodhart's Law of Image Verifiers**

A verifier that judges its own edits can report a pass when a blinded human
count disagrees. Human review is therefore required for defensible synthetic
CV data; `UNCERTAIN` is available for ambiguous mask cases, but its catch rate
in this audit was not logged and must not be presented as measured.

The single surviving image can be exported as a format demonstration in
COCO-shaped form with:

```bash
python -m scripts.export_coco
```

The saved experiment did not retain instance masks, so its annotation uses a
documented dummy full-image box. This is a format demonstration of the single
surviving image, not an instance-level training label. The metadata marks the image
`human_verified: true`, `artifact: false`, and
`mask_quality: "placeholder_full_frame"`.

**The 30% Artifact Problem**

The image model introduced visible seams or other artifacts in 7/21 audited
starting images (33.3%) and 25/84 edited images (29.8%). A numeric outcome
qualified as an **Accepted Clean Image** only when the human-verified count
matched the expected count and the human reported no visible artifact; this
occurred in 1/21 Numeric outcomes (4.8%). These audit results show why a
deterministic verifier must be paired with human review: it cannot establish
visual quality or ground-truth accuracy on its own.

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

### Human audit UI

With the project environment active, launch the blinded single-image audit app:

```bash
python -m scripts.audit_ui
```

It serves locally on `http://127.0.0.1:7861` and saves human counts and artifact
checks to `runs/audit/master_ground_truth.csv`.

## Evaluation

The live evaluation harness supports one-shot, blind random retries, pass@4, and feedback-driven loop modes:

```bash
python -m src.eval.run_eval --mode one-shot --max-cost 4.00 --live
python -m src.eval.run_eval --mode random_retry --max-cost 4.00 --live
python -m src.eval.run_eval --mode pass@4 --max-cost 4.00 --live
python -m src.eval.run_eval --mode loop --max-cost 4.00 --live
```

The `--max-cost` guard estimates configured Image/SAM costs and excludes Spark costs. To run without network access or API credentials, omit `--live` to use the mock clients.

## Limitations

- Undercounts (for example, generating 7 apples when asked for 8) can cause the editor to overshoot when instructed to “add 1”. Only two undercount failures occurred in this experiment, so this result is not well-powered.
- The human audit found 8 false passes among 11 SAM pass predictions (72.7%). SAM count equality alone is therefore not a trustworthy human-verified acceptance criterion for this sample.
- Visible artifacts were reported for 25/84 edited images (29.8%); the numeric arm's artifact rate was 6/21 (28.6%). The accepted-clean rate for numeric outcomes was 1/21 (4.8%).
- The single-step CSV does not retain per-image `UNCERTAIN` verdict statuses, so the verifier's catch rate for the eight false passes is not measurable from this audit.
- SAM can occasionally false-fail touching or overlapping objects. The verifier can mark some large or overlapping-mask count cases `UNCERTAIN`; this does not independently establish the true object count or visual quality.
- Multi-round evaluation crashed and did not complete.
- Undercount starts (n=2) were excluded.
- Hard prompts only.

Run the test suite with:

```bash
python -m pytest -q
```
