# Multi-stage research development rules

These rules apply to all RVQ trajectory and probing development tasks.

## Before implementation

For every development stage:

1. Read this `AGENTS.md`, the repository `README.md`, and files relevant to
   the requested stage.
2. Inspect `git status` and preserve existing uncommitted changes.
3. Compare the current implementation against the requested stage.
4. Propose an implementation plan before editing files.
5. Do not begin implementation until the user confirms the plan.

## Scope control

- Implement only the requested stage.
- Do not perform unrelated refactoring.
- Do not silently modify dataset labels, clinical definitions, speaker
  inclusion rules, or data splits.
- Preserve backward compatibility when possible.
- If backward compatibility is impossible, explain the reason before changing it.
- Do not use destructive Git commands.
- Do not overwrite unrelated user changes.

## Research validity

- Clinical probes must use speaker-disjoint evaluation.
- Speaker identity and clinical tasks must not automatically share the same split.
- Never treat utterances from the same speaker as independent clinical subjects.
- Do not describe ASR WER as clinical intelligibility.
- Do not claim clinical diagnosis or utterance-level clinical severity from
  speaker-level labels.
- Do not invent missing citations, labels, metadata, or experimental results.
- Clearly distinguish discrete learned embeddings from frozen codec-native
  embeddings.
- Token IDs are indices and must never be added or averaged as numeric features.

## Data and artifacts

Do not commit:

- TORGO or other restricted dataset audio;
- extracted codec tokens;
- codec checkpoints;
- trained model checkpoints;
- reconstructed audio collections;
- full experiment outputs;
- credentials or machine-specific absolute paths.

Use small synthetic fixtures or mocks for tests whenever real data, GPU,
checkpoints, or restricted datasets are unavailable.

## Testing

For every implementation stage:

1. Add or update relevant unit tests.
2. Run the relevant test suite.
3. Run a minimal smoke test when possible.
4. Report tests that could not be run and explain why.
5. Do not claim that full experiments succeeded unless they were actually run.

## Completion report

At the end of every implementation stage, provide:

1. changed files;
2. purpose of each change;
3. tests and smoke tests run;
4. test results;
5. work blocked by missing data, checkpoints, GPU, or dependencies;
6. example commands for the implemented feature;
7. known limitations;
8. recommended next stage.

Before committing:

1. inspect `git diff` and `git status`;
2. confirm that no dataset, checkpoint, token, audio, or large output file
   has been added;
3. suggest a concise commit message;
4. do not push unless explicitly requested.
