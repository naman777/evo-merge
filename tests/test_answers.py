import pytest

from evomerge.answers import extract_gold, extract_prediction, is_correct


@pytest.mark.parametrize("solution,gold", [
    ("जेनेट बेचती है 16-3-4 = 9 ... 18. ####18 ", 18.0),
    ("... 70,000 ###70000 ", 70000.0),
    ("कुल 3 बोल्ट ####3 ", 3.0),
    ("उत्तर #### 1,250", 1250.0),
    ("कोई उत्तर नहीं", None),
])
def test_extract_gold(solution, gold):
    assert extract_gold(solution) == gold


@pytest.mark.parametrize("output,pred", [
    ("So she makes 9 * 2 = 18 dollars. \\boxed{18}", 18.0),
    ("Answer: \\boxed{\\$1,250}", 1250.0),
    ("\\boxed{\\frac{7}{2}} wait, final \\boxed{3.5}", 3.5),
    ("अतः उत्तर १८ है", 18.0),
    ("The total is 80,000 + 50,000 = 130,000", 130000.0),
    ("no number here", None),
])
def test_extract_prediction(output, pred):
    assert extract_prediction(output) == pred


def test_is_correct():
    assert is_correct(18.0, 18.0)
    assert is_correct(18.00001, 18.0)
    assert not is_correct(17.0, 18.0)
    assert not is_correct(None, 18.0)
