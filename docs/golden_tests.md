# Golden tests

Golden tests compare `justdata` with pinned external reference implementations.
They certify numerical compatibility that ordinary shape and smoke tests cannot
establish.

## Certification levels

- **Declared**: the intended external contract is documented, but no checked-in
  oracle proves it.
- **Frontend Golden**: a passing checked-in fixture proves waveform-to-feature
  compatibility for the named preset.
- **Logits Golden**: a passing checked-in fixture also proves converted
  checkpoint logits. This requires reviewed checkpoint and derived-output
  redistribution terms.

A preset must not advertise a level higher than its passing checked-in test.
Frontend certification does not imply checkpoint or logits compatibility.

## Fixture policy

Golden fixtures must satisfy all of these rules:

- The oracle is produced by a pinned external implementation, never by
  `justdata`.
- Inputs are deterministic synthetic waveforms from
  `tests/acoustic/golden/corpus.npz`; no copyrighted source audio is stored.
- Each compressed fixture is at most 2 MiB, and the complete acoustic golden
  directory should remain below 5 MiB.
- Metadata records schema version, source repository and 40-character revision,
  source path, license, dependency versions, tensor layout, hashes, tolerances,
  and a numerical-tolerance rationale.
- Metadata contains no generation timestamp. Regenerating the waveform recipe
  must produce byte-identical arrays and metadata.
- A frontend or preset-hash semantic change requires external regeneration and
  recertification. Updating an oracle solely from changed `justdata` output is
  forbidden.

Each family README contains its regeneration command. First regenerate the
shared corpus when the waveform recipe intentionally changes:

```bash
python tests/acoustic/golden/generate_corpus.py
```

## Acoustic certification index

| Family | Frontend status | Logits status | External authority |
|---|---|---|---|
| AST | Frontend Golden | Declared | YuanGongND/ast + Torchaudio Kaldi fbank |
| EfficientAT / DyMN | Frontend Golden | Declared | fschmid56/EfficientAT `AugmentMelSTFT` |
| PaSST | Frontend Golden | Declared | kkoutini/PaSST `AugmentMelSTFT`, evaluation without Patchout |
| CED Hugging Face | Frontend Golden | Declared | `mispeech/ced-base` `CedFeatureExtractor` |
| CED ONNX/Kaldi helper | Declared | Declared | Deferred because its non-centered frontend disagrees with the Hugging Face extractor |
| PANNs Cnn14 16/32 kHz | Frontend Golden | Declared | qiuqiangkong/audioset_tagging_cnn + TorchLibrosa |

Logits remain Declared because this repository does not redistribute checkpoints
or checkpoint-derived output fixtures. The golden tests never download weights.

## Running the suites

Default development and CI remain independent of the golden stack:

```bash
uv run pytest
```

Install and run the optional reference suite with:

```bash
uv sync --extra golden
uv run pytest -o addopts= -m golden --junitxml=golden.xml
python tests/acoustic/golden/check_report.py golden.xml --expected 23
```

The report guard requires exactly 23 tests and zero skips. The GitLab golden job
is manual because of dependency cost, but it is not allowed to fail.

Vision has non-external preset contract tests in
`tests/test_vision_preset_contracts.py`; no vision preset currently claims a
golden certification level.
