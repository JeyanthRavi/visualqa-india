import torch

from visualqa.analyzer import select_answer_tokens
from visualqa.scoring import DIAGNOSTICS, classify_answer, score_answers


def _answers(default: str = "No, none is visible.") -> dict[str, str]:
    return {diagnostic.key: default for diagnostic in DIAGNOSTICS}


def test_negated_pothole_is_not_a_risk() -> None:
    diagnostic = DIAGNOSTICS[0]
    assert classify_answer("No potholes or cracks are visible.", diagnostic)[0] is False


def test_plural_pothole_without_yes_is_a_risk() -> None:
    diagnostic = DIAGNOSTICS[0]
    assert classify_answer("Several potholes are visible.", diagnostic)[0] is True


def test_unsafe_does_not_count_as_safe() -> None:
    diagnostic = DIAGNOSTICS[-1]
    is_risk, evidence = classify_answer("Yes, it looks unsafe.", diagnostic)
    assert is_risk is True
    assert evidence == "unsafe"


def test_each_category_is_deducted_only_once() -> None:
    answers = _answers()
    answers["road_damage"] = "Yes, potholes, cracks and a collapsed road are visible."
    report = score_answers(answers)
    assert report["score"] == 75
    assert len(report["findings"]) == 1


def test_all_risks_can_reach_zero() -> None:
    report = score_answers(_answers("Yes, the harmful condition is visible."))
    assert report["score"] == 0
    assert report["severity"] == "HIGH OBSERVED RISK"


def test_decoder_only_output_drops_prompt_tokens() -> None:
    output_ids = torch.tensor([[10, 11, 12, 20, 21]])
    answer_ids = select_answer_tokens(output_ids, prompt_length=3, decoder_only=True)
    assert answer_ids.tolist() == [[20, 21]]


def test_encoder_decoder_output_is_not_sliced() -> None:
    output_ids = torch.tensor([[20, 21]])
    answer_ids = select_answer_tokens(output_ids, prompt_length=3, decoder_only=False)
    assert answer_ids.tolist() == [[20, 21]]


def test_specialist_detector_overrides_blip_road_answer() -> None:
    answers = _answers()
    answers["road_damage"] = "No, the road looks smooth."
    report = score_answers(
        answers,
        risk_overrides={"road_damage": (True, "YOLO detected a pothole")},
    )
    assert report["score"] == 75
    assert report["findings"][0]["source"] == "specialist detector"


def test_negative_detector_override_prevents_blip_false_positive() -> None:
    answers = _answers()
    answers["road_damage"] = "Yes, potholes are visible."
    report = score_answers(
        answers,
        risk_overrides={"road_damage": (False, "No detection above threshold")},
    )
    assert report["score"] == 100
