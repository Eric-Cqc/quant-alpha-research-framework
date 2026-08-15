from signal_research.signal_generation.registry import build_candidate_grid
from signal_research.validation.dedup import canonical_signature, remove_exact_duplicates


def test_exact_duplicates_are_removed_with_record():
    specs = build_candidate_grid()
    valid_like = [spec for spec in specs if spec.name.startswith("reversal_5")]
    kept, report = remove_exact_duplicates(valid_like)
    assert len(kept) == 1
    assert len(report) == 2
    assert canonical_signature(report[0] and kept[0]) == report[1].signature
    assert report[1].kept is False
