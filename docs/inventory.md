# Strict local inventory admission (JD-01)

`justdata.core.admit_inventory` validates a finite, resolved inventory before returning a usable dataset. It reads local
assets, checks their full SHA-256 digests, runs the caller's reader and optional adapter/validator, and writes a
self-contained disk snapshot. No TFDS or Hugging Face source is resolved or downloaded by this route. Readers and
callbacks are trusted caller code and must also operate offline.

Existing `fetch_ds` and `load_ds` retain their legacy behavior and return types. They can skip failed sources and do
**not** provide this admission guarantee. The new inventory API is always strict and has no permissive mode.

## Inputs and ordering

All public types and functions below are exported from `justdata.core`:

| Interface                                                                                                | Contract                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `InventoryAsset(path, sha256)`                                                                           | Local file and its expected full SHA-256 hexadecimal digest.                                                                              |
| `InventoryRecord(record_id, assets={}, metadata={})`                                                     | Nonempty opaque string ID, named assets, and JSON-compatible reader metadata. IDs must be unique across all requested sources and splits. |
| `InventorySource(splits, reader, adapter=None, validate=None)`                                           | Explicit split mappings to finite record lists/tuples; one record reader; optional adapter and domain validator.                          |
| `InventoryFilter(transformation_id, predicate)`                                                          | Named selection returning a scalar boolean for each canonical sample.                                                                     |
| `admit_inventory(sources, splits_info, *, inventory_id, snapshot_dir, output_signature, selection=None)` | Validate and snapshot the complete request; return an `AdmittedInventory`.                                                                |
| `open_inventory(snapshot_dir)`                                                                           | Verify and reopen a completed snapshot without the original sources or callbacks.                                                         |
| `load_inventory(admitted, dataset_type, batch_size, seed, **pipeline_options)`                           | Run the existing preprocessing and finalization machinery on an admitted dataset.                                                         |

`sources` maps names to `InventorySource` objects. `splits_info` is an ordered mapping of required source names to
nonempty split lists. Admission preserves that mapping's order, each requested split's order, and each split's record
order. Missing sources, missing splits, empty split **declarations**, duplicate requested splits, and duplicate IDs
fail. A declared split containing an empty record list is intentionally empty and succeeds, including when the complete
inventory is empty. A tensor signature is required even for empty data.

The reader receives `(payloads, metadata)`: a mapping from asset role to verified `bytes`, and a detached copy of the
record's metadata. It returns one sample dictionary. It must eagerly decode and validate all external content needed by
the sample, using the supplied bytes rather than reopening original paths. The library verifies the bytes passed to the
reader; it cannot audit arbitrary I/O hidden inside caller code. Deferred path-only media samples are outside this
contract. In-memory fixtures can use records without file assets.

`adapter=None` explicitly selects an already canonical reader result. A callable adapter runs eagerly; a string selects
a required registered adapter using the existing exact/longest-prefix lookup. An unknown name fails instead of falling
back to identity. `get_adapter(name, required=True)` also exposes that lookup directly; its default behavior is
unchanged.

`output_signature` is a nested dictionary of known-rank `tf.TensorSpec` leaves; dimensions may be `None`. Numeric,
boolean, and string tensors are supported. It describes the sample after adaptation. Tensor dtypes and shapes are
checked without numeric casts. The library adds a scalar string `metadata.example_id` to both the signature and records,
preserving the complete supplied ID. A conflicting reader or adapter ID fails. Other metadata must be included in the
signature and is preserved as supplied by the reader/adapter.

The optional `validate(sample)` callback checks domain rules after adaptation and tensor validation. It must return
`None` and raise on invalid data. This is where a product can validate categorical IDs or annotation coverage; the core
does not prescribe modality fields or a LaRS ontology.

## Runnable offline image/mask example

This example creates synthetic local fixtures and their trusted inventory, then admits and reopens them. For real data,
use expected digests from the previously resolved immutable inventory; recomputing expected digests from changed files
at admission would defeat change detection.

```python
import hashlib
import tempfile
from pathlib import Path

import tensorflow as tf

from justdata.core import (
    InventoryAsset,
    InventoryRecord,
    InventorySource,
    admit_inventory,
    load_inventory,
    open_inventory,
)


def read_pair(payloads, metadata):
    return {
        "image": tf.io.decode_png(payloads["rgb"], channels=3),
        "mask": tf.cast(
            tf.squeeze(tf.io.decode_png(payloads["annotation"], channels=1), -1),
            tf.int32,
        ),
    }


def identity(sample, **kwargs):
    return sample


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    assets = {}
    for role, tensor in {
        "rgb": tf.ones([8, 10, 3], tf.uint8),
        "annotation": tf.zeros([8, 10, 1], tf.uint8),
    }.items():
        content = tf.io.encode_png(tensor).numpy()
        path = root / f"{role}.png"
        path.write_bytes(content)
        assets[role] = InventoryAsset(path, hashlib.sha256(content).hexdigest())

    source = InventorySource(
        splits={"validation": [InventoryRecord("frame/revision-1", assets)]},
        reader=read_pair,
    )
    admitted = admit_inventory(
        {"reviewed-local": source},
        {"reviewed-local": ["validation"]},
        inventory_id="synthetic-inventory-v1",
        snapshot_dir=root / "admitted",
        output_signature={
            "image": tf.TensorSpec([None, None, 3], tf.uint8),
            "mask": tf.TensorSpec([None, None], tf.int32),
        },
    )
    reopened = open_inventory(admitted.path)
    batches, n_batches = load_inventory(
        reopened,
        dataset_type="validation",
        batch_size=2,
        seed=0,
        preprocess_fn=identity,
        postprocess_fn=identity,
        map_parallel_calls=1,
        private_threadpool_size=1,
    )
    batch = next(iter(batches))
    assert n_batches == 1
    assert batch["padding_mask"].numpy().tolist() == [True, False]
    assert reopened.report["retained_count"] == 1
```

## Selection, evidence, and failures

Pass `selection=InventoryFilter("selection-rule-v1", predicate)` to declare intentional filtering. Every requested
record is read, decoded, adapted, and validated **before** selection. Exclusion cannot hide missing assets or invalid
records. A filter exception or a non-scalar/non-boolean result fails admission. A filter retaining zero records is a
successful, explicitly empty result.

`admitted.report` is JSON-serializable and includes:

- `schema`, `inventory_id`, `complete`, and the selection's identifier bound to that inventory;
- requested, retained, and excluded counts, plus ordered IDs for each source/split;
- ordered per-record evidence with disposition, source, split, reader metadata, asset paths, verified full digests, and
  byte counts;
- `snapshot` digests for the TFRecord and tensor schema, and `failures`.

Every requested ID must appear exactly once as retained or explicitly excluded. The dataset contains retained IDs in
their original order. Admission rereads the written snapshot to check actual IDs, tensor parsing, and cardinality before
publishing it. An `InventoryLoadError` supplies a detached partial report with `complete=False` and an `issue`
dictionary; `error.to_dict()` returns both.

Stable issue codes are `invalid_declaration`, `invalid_splits`, `missing_source`, `missing_split`, `duplicate_id`,
`unknown_adapter`, `invalid_schema`, `invalid_record`, `asset_read`, `digest_mismatch`, `reader_failed`,
`adapter_failed`, `filter_failed`, `snapshot_exists`, `snapshot_io`, `snapshot_digest_mismatch`, and `invalid_snapshot`.
Available context includes `source`, `split`, `record_id`, asset role, path, and expected/actual digests. The original
exception is retained as `__cause__`. This includes TensorFlow decoding errors raised during iteration inside a reader.
No log parsing is needed.

## Snapshot and pipeline lifecycle

The snapshot destination must not exist and its parent must exist. The caller owns its storage and cleanup. Admission
never replaces an existing snapshot. It streams one record at a time to an uncompressed TFRecord; inventory metadata and
identity evidence remain in memory. Decoded media can require substantially more disk space than source files.

Snapshots contain `records.tfrecord`, `schema.json`, `report.json`, and a final `manifest.json` with full artifact
digests and format version `justdata.inventory.v1`. The manifest is the atomic completion marker. Readers reject missing
or incomplete manifests. A caught failure removes only the destination created by that admission. An abrupt process
termination may leave an incomplete directory, which must be inspected and removed before retry. Opening verifies
digests and fully scans snapshot records, without executing serialized Python objects or needing the original reader.

After admission, changing or removing original source files does not affect the snapshot. Readmission with stale asset
digests fails. Keep snapshot directories immutable while datasets use them: verification occurs at admission/opening,
not before every subsequent batch read. Digests provide consistency relative to the local manifest, not proof of origin
against replacement of the entire package.

`load_inventory` accepts the existing `load_ds` pipeline, caching, execution, metadata, NumPy, and batching controls,
with the same normal and `return_raw_ds=True` results, including `finalize_epoch`. Source arguments are absent, and both
`source_filter_fn` and `filter_fn` are rejected; selection must be declared at admission. Further manual dataset
filtering creates a different population outside the admission report.

Report counts describe admitted source records, independently of training shuffle, view expansion, dropped batches, or
padded rows. `metadata_mode` retains its existing behavior: `numeric_only` removes string IDs and `none` removes
metadata. Full numeric identity transport, sidecar persistence, geometry, and resume qualification belong to subsequent
JD requirements; this API does not claim those guarantees.
