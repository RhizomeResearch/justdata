# PaSST golden fixture

This fixture certifies deterministic evaluation preprocessing only. Patchout is
a training augmentation and is deliberately absent from the oracle. The source
waveform is the repository's synthetic CC0 corpus.

```bash
python tests/acoustic/golden/generate_corpus.py
python tests/acoustic/golden/passt/generate_passt_golden.py \
  --reference-root /path/to/PaSST
```

Checkpoint logits remain Declared because no compact checkpoint-derived oracle
with reviewed redistribution terms is checked in.
