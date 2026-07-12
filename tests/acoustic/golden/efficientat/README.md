# EfficientAT/DyMN golden fixture

This fixture certifies the shared EfficientAT/DyMN evaluation frontend against
the MIT-licensed upstream `AugmentMelSTFT` implementation. It contains only
features derived from the repository's synthetic CC0 waveform corpus.

Regenerate from the pinned checkout:

```bash
python tests/acoustic/golden/generate_corpus.py
python tests/acoustic/golden/efficientat/generate_efficientat_golden.py \
  --reference-root /path/to/EfficientAT
```

Checkpoint logits remain Declared: no checkpoint or checkpoint-derived output
is redistributed.
