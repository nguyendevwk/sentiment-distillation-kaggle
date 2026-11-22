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


# SentiBench official known aspect categories
ASQP_CATEGORIES = {
    'ambience general', 'drinks prices', 'drinks quality', 'drinks style_options',
    'food general', 'food prices', 'food quality', 'food style_options',
    'location general', 'restaurant general', 'restaurant miscellaneous',
    'restaurant prices', 'restaurant pcrices', 'service general',
    'battery miscellaneous', 'battery operation_performance', 'battery quality',
    'company general', 'cpu design_features', 'cpu miscellaneous',
    'cpu operation_performance', 'cpu quality', 'display design_features',
    'display general', 'display operation_performance', 'display quality',
    'display usability', 'fans_cooling design_features', 'fans_cooling operation_performance',
    'fans_cooling quality', 'graphics design_features', 'graphics general',
    'graphics miscellaneous', 'graphics quality', 'hard_disc design_features',
    'hard_disc quality', 'hardware general', 'hardware operation_performance',
    'hardware quality', 'keyboard design_features', 'keyboard general',
    'keyboard operation_performance', 'keyboard quality', 'keyboard usability',
    'laptop connectivity', 'laptop design_features', 'laptop general',
    'laptop miscellaneous', 'laptop operation_performance', 'laptop portability',
    'laptop price', 'laptop quality', 'laptop usability', 'memory design_features',
    'motherboard quality', 'mouse design_features', 'mouse general',
    'mouse operation_performance', 'mouse usability', 'multimedia_devices design_features',
    'multimedia_devices general', 'multimedia_devices operation_performance',
    'multimedia_devices price', 'multimedia_devices quality', 'multimedia_devices usability',
    'optical_drives design_features', 'optical_drives operation_performance',
    'optical_drives quality', 'os design_features', 'os general',
    'os operation_performance', 'os usability', 'out_of_scope design_features',
    'out_of_scope general', 'out_of_scope operation_performance', 'out_of_scope usability',
    'ports design_features', 'ports general', 'ports operation_performance',
    'ports quality', 'power_supply design_features', 'power_supply general',
    'power_supply operation_performance', 'power_supply quality', 'shipping quality',
    'software design_features', 'software general', 'software operation_performance',
    'software usability', 'support design_features', 'support general',
    'support operation_performance', 'support price', 'support quality', 'warranty general'
}


def asqp_prediction(text, sentence):
    raw, result, valid = literal_tuples(text, 4)
    normalized = []
    for r in result:
        cat, asp, opn, pol = r[0], r[1], r[2], r[3]
        # Align polarity to index 3 if model emitted (aspect, category, polarity, opinion)
        if str(opn).strip().lower() in ('positive', 'negative', 'neutral') and str(pol).strip().lower() not in ('positive', 'negative', 'neutral'):
            opn, pol = pol, opn
        # Align category to index 0 if model emitted prompt order (aspect term, aspect category, ...)
        cat_lower = str(cat).strip().lower()
        asp_lower = str(asp).strip().lower()
        if asp_lower in ASQP_CATEGORIES and cat_lower not in ASQP_CATEGORIES:
            cat, asp = asp, cat
        normalized.append((cat, asp, opn, str(pol).strip().lower()))
    kept = [r for r in normalized if r[3] in ('positive', 'negative', 'neutral')]
    return normalized, kept, valid and len(kept) == len(normalized)


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
