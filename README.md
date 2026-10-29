# image-unit-tests

image-unit-tests is a deterministic image-generation agent designed to turn a user brief into structured constraints, generate an image, verify it with SAM masks and pure Python logic, and iterate until it passes or reaches the configured round limit.

Built during Meta Global AI Developer Hackathon (Indie track, 'Push the Frontier').

IMPORTANT_RULES_AND_RESSOURCES.md (rules, links, docs) and .env (secrets) must exist in the repo root. The .env file is git-ignored; contributors should copy .env.example to .env before running any real API work.

Architecture diagram:

+--------------------+    +--------------------+    +--------------------+
| User brief         | -> | SparkClient        | -> | Constraints JSON   |
+--------------------+    +--------------------+    +--------------------+
          |                                   |
          v                                   v
+--------------------+    +--------------------+    +--------------------+
| ImageClient        | -> | SAM masks          | -> | Verifier           |
+--------------------+    +--------------------+    +--------------------+
          |                                   |
          +-------------------> Agent loop <------------------+
                               |
                               v
                      Fix instructions / logging

Configure `config.yaml` and `.env` by setting the Meta or Baseten base URLs, API keys, and model IDs that already exist in the workspace. The example file records the expected env names without exposing secrets.

Run tests:

```bash
pytest
```

Mock mode:

```bash
python -m src.eval.run_eval --mode loop --max-cost 2.00
```

Live mode:

```bash
python -m src.eval.run_eval --mode loop --max-cost 2.00 --live
```

To run the Gradio UI:

```bash
python -m src.ui.app
```

The project includes deterministic mock clients so the pipeline can run without credits or network access.
