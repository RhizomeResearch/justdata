# CED golden fixture

The Apache-2.0 Hugging Face `CedFeatureExtractor` is the certified authority.
Its mel-spectrogram and amplitude-to-dB calls are reproduced verbatim in the
generator without importing `justdata`.

```bash
python tests/acoustic/golden/generate_corpus.py
python tests/acoustic/golden/ced/generate_ced_golden.py
```

The upstream ONNX/Kaldi helper is left Declared because its non-centered
frontend disagrees with the Hugging Face extractor. Logits are also Declared.
