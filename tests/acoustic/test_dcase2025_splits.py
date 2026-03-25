from pathlib import Path

import pytest
import tensorflow as tf

from justdata.acoustic.dcase2025 import (
    SplitLeakageError,
    assert_stats_allowed,
    load_dcase2025_task1_splits,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dcase2025"


def test_dev_train_allows_stats():
    assert_stats_allowed("dev_train_25")


def test_dev_test_rejects_stats():
    with pytest.raises(SplitLeakageError):
        assert_stats_allowed("dev_test")


def test_eval_rejects_stats():
    with pytest.raises(SplitLeakageError):
        assert_stats_allowed("eval")


def test_override_stats_emits_warning():
    with pytest.warns(RuntimeWarning):
        assert_stats_allowed("dev_test", allow_override=True)


def test_dcase_split_cardinality_fixture():
    train, dev_test, eval_ds = load_dcase2025_task1_splits(
        "dcase2025_task1",
        ["dev_train_25", "dev_test", "eval"],
        FIXTURE_DIR,
    )

    assert tf.data.Dataset.cardinality(train).numpy() == 4
    assert tf.data.Dataset.cardinality(dev_test).numpy() == 2
    assert tf.data.Dataset.cardinality(eval_ds).numpy() == 2
