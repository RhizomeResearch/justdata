# PANNs golden fixtures

These fixtures use the exact MIT-licensed TorchLibrosa `Spectrogram` and
`LogmelFilterBank` components instantiated by upstream Cnn14, at both official
32 kHz and 16 kHz settings.

```bash
python tests/acoustic/golden/generate_corpus.py
python tests/acoustic/golden/panns/generate_panns_golden.py
```

Only frontend features from synthetic CC0 waveforms are checked in. Logits
remain Declared.
