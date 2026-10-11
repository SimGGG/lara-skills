# Contributing

Thank you for your interest. Issues and pull requests are welcome.

- Run the tests before opening a pull request: `pip install -e ".[dev]"`, then `pytest` and `lara-skills quickstart --workspace ci_workspace --device cpu`.
- Keep the research semantics intact: changes to the scoring rule, the training objective, checkpoint selection or the metrics need a note in `CHANGELOG.md` and should stay optional behind configuration keys.
- Do not add real posting, dictionary, patent or judgment data to the repository. The toy data are synthetic; regenerate them with `lara-skills toy-data` and update `examples/toy/toy_manifest.json` when the generator changes.
- Every pipeline stage must keep writing a manifest with input hashes and must refuse to overwrite existing outputs.
