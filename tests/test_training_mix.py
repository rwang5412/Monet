"""--num_samples is a head slice of the concatenated data files; Visual_CoT alone
is 118K rows, so an unshuffled 32K pilot trains on ONE dataset. These cover the
per-file tail holdout and the mix report that make that visible."""
from src.utils import drop_tail, dataset_mix


def test_drop_tail():
    rows = list(range(10))
    assert drop_tail(rows, 0) == rows
    assert drop_tail(rows, -1) == rows
    assert drop_tail(rows, 3) == list(range(7))
    assert drop_tail(rows, 10) == []
    assert drop_tail(rows, 99) == []


def test_dataset_mix_counts_and_orders_by_size():
    rows = ([{"metadata": {"dataset_name": "Visual_CoT"}}] * 5
            + [{"metadata": {"dataset_name": "CogCoM"}}] * 2
            + [{"metadata": {}}])
    assert dataset_mix(rows) == {"Visual_CoT": 5, "CogCoM": 2, "?": 1}


def test_head_slice_without_shuffle_is_one_dataset():
    # the failure mode itself, stated as a test
    files = [[{"metadata": {"dataset_name": "Visual_CoT"}}] * 100,
             [{"metadata": {"dataset_name": "Zebra"}}] * 5]
    concat = [r for f in files for r in f]
    assert len(dataset_mix(concat[:32])) == 1
