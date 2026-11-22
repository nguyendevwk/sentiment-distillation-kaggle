"""Safe, non-generative parser shared by all corrected FSA evaluators.

Accepts lists/tuples, multiline literals, code fences and an optional Label:
prefix. Never fabricates closing brackets, cuts rows to three or swaps fields.
"""
import ast


def literal_tuples(text, width):
    raw = text
    text = text.strip()
    if text.startswith('```'):
        parts = text.splitlines()
        if len(parts) >= 2 and parts[-1].strip() == '```':
            text = '\n'.join(parts[1:-1]).strip()
    if text.startswith('Label:'):
        text = text[len('Label:'):].strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return raw, [], False
    if not isinstance(parsed, (list, tuple)):
        return raw, [], False
    if len(parsed) == width and all(isinstance(v, str) for v in parsed):
        parsed = [parsed]
    result, valid = [], True
    for row in parsed:
        if not isinstance(row, (list, tuple)) or len(row) != width or not all(isinstance(v, str) for v in row):
            valid = False
            continue
        # Preserve span text: official gold includes significant leading spaces.
        value = tuple(row)
        if value not in result:
            result.append(value)
    return raw, result, valid


def asqp_prediction(text, sentence):
    raw, result, valid = literal_tuples(text, 4)
    kept = [r for r in result if r[3] in ('positive', 'negative', 'neutral')]
    return result, kept, valid and len(kept) == len(result)


def ssa_prediction(text, sentence):
    raw, result, valid = literal_tuples(text, 4)
    kept = [r for r in result if r[3] in ('positive', 'negative', 'neutral')]
    non_null = [r for r in result if r != ('NULL', 'NULL', 'NULL', 'NULL')]
    # Author graph scorer consumes the first return value, not only `kept`.
    return kept, kept, valid and len(kept) == len(non_null)


def atsa_prediction(text, sentence):
    _, result, _ = literal_tuples(text, 2)
    return [r for r in result if r[0] in sentence and r[1] in ('positive', 'negative', 'neutral', 'conflict')]


def acsa_prediction(text):
    _, result, _ = literal_tuples(text, 2)
    return [r for r in result if r[1] in ('positive', 'negative', 'neutral')]
