"""Pulling numeric answers out of model outputs and Hindi GSM8K reference solutions."""
import re

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(text):
    text = text.translate(DEVANAGARI_DIGITS).replace(",", "").replace(" ", "")
    match = NUMBER.search(text)
    return float(match.group()) if match else None


def _last_number(text):
    text = text.translate(DEVANAGARI_DIGITS)
    # Drop thousands separators ("80,000") but keep list commas ("1, 2").
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    matches = NUMBER.findall(text)
    return float(matches[-1]) if matches else None


def _last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def extract_prediction(output):
    """The model's final answer: the last \\boxed{...} if any, else the last number."""
    boxed = _last_boxed(output)
    if boxed is not None:
        value = _to_float(boxed)
        if value is not None:
            return value
    return _last_number(output)


def extract_gold(solution):
    """Reference answer after the '####' marker (the Hindi translation sometimes has '###')."""
    markers = re.findall(r"#{2,}\s*([-\d,.\s०-९]+)", solution)
    if markers:
        value = _to_float(markers[-1])
        if value is not None:
            return value
    return None


def is_correct(prediction, gold, tol=1e-4):
    return prediction is not None and gold is not None and abs(prediction - gold) <= tol * max(1.0, abs(gold))
