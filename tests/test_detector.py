from visualqa.detector import resolve_yolo_model_path


def test_explicit_detector_path_is_resolved(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test checkpoint placeholder")
    assert resolve_yolo_model_path(checkpoint) == checkpoint.resolve()


def test_missing_explicit_detector_returns_none(tmp_path) -> None:
    assert resolve_yolo_model_path(tmp_path / "missing.pt") is None
