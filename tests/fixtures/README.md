# Mini dataset fixture (P1 + P2) for fast offline robustness tests.

Regenerate from the open dataset:

```bash
python scripts/build_mini_fixture.py
```

Record LLM replay responses (requires `OPENAI_API_KEY`, run once after pipeline changes):

```bash
python scripts/record_mini_llm_fixtures.py
```

Replay fixtures are stored in `tests/fixtures/llm/` as `{prompt_hash}.json`.
