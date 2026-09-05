# Performance checks

Run the standalone CPU benchmark from the repository root:

```bash
uv run python benchmarks/benchmark_performance.py --repeats 5
```

The benchmark uses synthetic records and temporary PNG/WAV files, requires no
downloads, and removes its fixtures when finished. It fixes TensorFlow inter-op,
intra-op, and private dataset pools to two threads. Output is JSONL with separate
construction and epoch timings, warmed operation timings, and output hashes.
Hashing is outside timed sections; temporary fixture directory names are
normalized when hashing metadata. Allocation figures use `tracemalloc` and
exclude TensorFlow's native allocations.

To compare against a saved source tree using the same installed dependencies:

```bash
PYTHONPATH=/path/to/baseline/src uv run python benchmarks/benchmark_performance.py --repeats 5
```

Compare output hashes before interpreting timing differences. Repeat comparisons
on an otherwise idle machine; CPU scheduling and frequency changes affect small
timing differences. These measurements do not establish GPU performance.

## September 2026 audit

Compared with revision `355465b0`, using Python 3.12.5, TensorFlow 2.21.0 with
oneDNN enabled, and an AMD Ryzen 9 7940HS CPU. All 11 benchmark output hashes
matched. Exact-parity tests additionally cover source schemas, empty splits,
iteration-time errors, repeated decoding, view metadata, dynamic shapes, and
input ownership.

Five-repeat standalone medians:

| Workload | Original | Updated |
| --- | ---: | ---: |
| 10,000 local audio records, epoch without decoding | 3,934 ms | 24 ms |
| 10,000 DCASE records, epoch without decoding | 4,958 ms | 47 ms |
| 257 PNG records, including decoding | 176 ms | 89 ms |
| 257-image evaluation pipeline | 419 ms | 334 ms |

Materialized audio records now incur an upfront tensor conversion: construction
rose from 21 to 62 ms for 10,000 local records and from 26 to 96 ms for DCASE.
The epoch savings exceed this setup cost even on the first pass. Only records
are tensorized; media are still read and decoded during iteration. Conversion
failures retain the original generator's iteration-time error behavior.

An initial standalone audio-pipeline timing was noisy. Ten paired rounds,
alternating original and updated callables in one process, confirmed the
following medians:

| Workload | Original | Updated |
| --- | ---: | ---: |
| One 224-pixel evaluation crop | 350 µs | 297 µs |
| Mono convolution, 4,096 samples and five taps | 244 µs | 210 µs |
| HF waveform conversion, 320,000 stereo samples | 566 µs | 331 µs |
| Statistics, 32,000 × 128 float64 observations | 12.37 ms | 7.79 ms |
| 257-clip WAV-to-logmel evaluation pipeline | 307 ms | 252 ms |

Unchanged five-view and stereo-convolution controls were within 2% in the paired
comparison. Audio-pipeline construction was 445 versus 422 ms. Statistics peak
temporary allocations fell from 62.5 to 31.3 MiB; waveform conversion fell from
7.33 to 4.88 MiB.

The implementation preserves arithmetic and accumulation order. It removes
Python callbacks for materialized records, bypasses one-item tensor loops, and
reuses disposable buffers. Loader stage order, RNG assignment, concurrency
defaults, preset hashes, frontend kernels, and source validation are unchanged.

## Training follow-up

Run the training benchmarks separately:

```bash
uv run python benchmarks/benchmark_training.py --repeats 5
PYTHONPATH=/path/to/baseline/src uv run python benchmarks/benchmark_training.py --repeats 5
```

These cases cover spectrogram masking with retained waveforms, small and large
patchout grids, NCHW/NHWC batch mixing, random erasing, solarization, metadata
projection, and complete seeded training epochs. TensorFlow inter-op and
intra-op pools use two threads; dataset pools use four, with one intra-op thread
per dataset operation. Each case reports an output hash outside the timed loop.
Older source trees without `augment_is_stateless` run both epoch cases serially.

The fast paths retain the existing arithmetic and RNG streams. Spectrogram
masking forwards unchanged batch fields while preserving shape validation;
patchout uses scattering when the pairwise mask would exceed 64 KiB. Vision
mixing preserves the public layout heuristic and uses NCHW kernels when erasing
is disabled. Random erasing broadcasts a single spatial mask across channels,
and solarization avoids pixel arithmetic when its gate is off.

Metadata projection is fused only when discarded values are passed through or
also returned elsewhere. Computed discarded values retain the original map
boundary so TensorFlow cannot suppress their errors. Sidecar writers and
postprocessing hooks also retain their stage boundaries. Parallel epoch
augmentation requires the caller's explicit stateless guarantee, described in
the [epoch API](../README.md#reusable-explicitly-seeded-epochs).

Compared with the source snapshot after the first audit (`bf2cf41e`), on the
same CPU/software described above, all 18 training output hashes matched.
The original benchmark's 11 hashes also matched, including both modality
pipelines. Five-repeat standalone medians:

| Workload | Before | After |
| --- | ---: | ---: |
| Spectrogram masking, 32 rows retaining 32,000-sample waveforms | 2.14 ms | 1.37 ms |
| Same stage retaining 320,000-sample waveforms | 13.12 ms | 7.57 ms |
| Patchout, 12,800 positions and 1,000 dropped positions | 3.20 ms | 1.18 ms |
| Mixup, NCHW batch of 32 × 224 × 224 RGB images | 10.34 ms | 7.18 ms |
| CutMix, same batch | 14.31 ms | 8.44 ms |
| Seeded 512-image training epoch, serial → four augmentation workers | 1,430 ms | 549 ms |

The unchanged serial epoch measured 1,427 ms after the changes. NHWC mixing
controls stayed within 2% in this comparison. Small patchout grids retain the
pairwise implementation.

Seven paired rounds alternating original and updated callables confirmed the
remaining gains:

| Workload | Before | After |
| --- | ---: | ---: |
| Random erasing, batch of 32, probability 0.25 | 8.02 ms | 7.48 ms |
| Random erasing, probability 1.0 | 16.06 ms | 13.91 ms |
| Solarization, batch of 32, probability 0.2 | 15.48 ms | 8.25 ms |
| Numeric metadata, 10,000 lightweight records | 89.41 ms | 67.50 ms |
| No metadata, same records | 85.29 ms | 73.42 ms |

Always-on solarization was within 2% in the paired comparison. Full-metadata
control timings varied in both directions between runs; no improvement is
claimed for that path. These figures measure CPU input-pipeline work, not
end-to-end model-training or GPU throughput.
