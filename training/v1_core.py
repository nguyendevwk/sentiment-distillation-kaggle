"""Dependency-free v1 normalization and deterministic budgeted selection.

Scores are ranking proxies, never certificates of teacher correctness.
"""
from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
import random
import re

from distill_core import digest


def iter_objects(path):
    """Stream a JSON array or JSONL. Malformed files stop the run."""
    path = Path(path)
    with path.open(encoding='utf-8-sig') as stream:
        first = ''
        while not first:
            char = stream.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
        if first != '[':
            stream.seek(0)
            for number, line in enumerate(stream, 1):
                if line.strip():
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(f'{path}:{number}: expected object')
                    yield number, obj
            return
        decoder, buffer, number, expecting_value = json.JSONDecoder(), '', 0, True
        eof = False
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                chunk = stream.read(65536)
                eof = not chunk
                buffer += chunk
                continue
            if buffer.startswith(']'):
                if expecting_value and number:
                    raise ValueError(f'{path}: trailing comma')
                if (buffer[1:] + stream.read()).strip():
                    raise ValueError(f'{path}: trailing content')
                return
            if not expecting_value:
                if not buffer.startswith(','):
                    raise ValueError(f'{path}: missing comma or closing bracket')
                buffer, expecting_value = buffer[1:], True
                continue
            try:
                obj, end = decoder.raw_decode(buffer)
            except ValueError:
                if eof:
                    raise ValueError(f'{path}: invalid or incomplete JSON array') from None
                chunk = stream.read(65536)
                eof = not chunk
                buffer += chunk
                continue
            if not isinstance(obj, dict):
                raise ValueError(f'{path}: expected object at index {number}')
            number += 1
            yield number, obj
            buffer, expecting_value = buffer[end:], False


def normalized_text(text):
    return ' '.join(text.split()).casefold()


def source_text(prompt, stage):
    """Conservative, logged extraction. Unknown templates are excluded, not guessed.

    KNOW: final Review/Text/Tweet section. ICL: last supported input marker,
    retaining auxiliary fields (e.g. Edit) and stripping only terminal Output.
    This isolates the final query, NOT texts inside demonstrations.
    """
    labels = ('Review|Text|Tweet' if stage == 'knowdist'
              else 'Sentence|Sentence1|Premise|Input|Text|Review|Tweet|News Headline')
    matches = list(re.finditer(rf'(?m)^(?:{labels}):[ \t]*', prompt))
    if not matches:
        raise ValueError('source_unresolved: unsupported input marker; inspect mapping')
    value = prompt[matches[-1].end():]
    if stage == 'icldist':
        value = re.sub(r'\n(?:Output|Label|Answer):[ \t]*\Z', '', value)
    if not value.strip():
        raise ValueError('source_unresolved: empty final input')
    return value


def canonical_record(obj, relative_file, index, source_sha):
    if 'prompt' in obj and 'llm_response' in obj:
        stage, prompt, response = 'knowdist', obj['prompt'], obj['llm_response']
        kind = obj.get('instr_type') or Path(relative_file).stem.removeprefix('sample_')
        if not str(kind).startswith(('analysis_', 'rewrite_')):
            raise ValueError('unknown KNOWDIST prompt type; supply instr_type')
    elif 'instruction' in obj and 'output' in obj:
        stage, response = 'icldist', obj['output']
        instruction, extra = obj['instruction'], obj.get('input', '')
        if not isinstance(instruction, str) or not isinstance(extra, str):
            raise ValueError('instruction/input must be strings')
        prompt = instruction + ('\n\n' + extra if extra.strip() else '')
        kind = ('task_diversification' if 'task_diversification' in relative_file
                else 'sentiment_alignment' if 'sentiment_distillation' in relative_file else None)
        if kind is None:
            raise ValueError('unknown ICLDIST file; cannot identify TD vs sentiment')
    else:
        raise ValueError('unknown_schema')
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
        raise ValueError('empty_or_nonstring_prompt_response')
    source = obj.get('source_text')
    method = 'explicit_source_text' if isinstance(source, str) and source.strip() else 'final_input_marker_v1'
    if method != 'explicit_source_text' and stage == 'icldist' and obj.get('input', '').strip():
        source, method = obj['input'], 'explicit_input_field'
    if method == 'final_input_marker_v1':
        source = source_text(prompt, stage)
    name = str(obj.get('dataset_name', 'unknown'))
    return {'id': f'{source_sha}:{index}', 'stage': stage, 'prompt': prompt,
            'response': response, 'source_key': digest(normalized_text(source)),
            'source_text': source, 'source_extraction': method,
            'group': f'{kind}:{name}', 'kind': kind, 'source_dataset': name,
            'file': relative_file, 'position': index}


def split_name(source_key, seed, dev_fraction):
    if not 0 < dev_fraction < 1:
        raise ValueError('dev_fraction must be in (0,1)')
    value = int(digest([seed, source_key])[:16], 16) / 2**64
    return 'probe' if value < dev_fraction / 2 else 'holdout' if value < dev_fraction else 'train'


def limited_candidates(rows, seed, per_group, caps):
    """Stable hash reservoirs; bounded memory per (stage,split,group).

    Train gets a total per-stage cap using round-robin group coverage. Dev has
    its own per-group cap, independent of the selection method.
    """
    if per_group < 1 or any(v < 1 for v in caps.values()):
        raise ValueError('candidate limits must be positive')
    buckets = defaultdict(list)
    for r in rows:
        key = (r['stage'], r['split'], r['group'])
        limit = per_group if r['split'] == 'train' else 8
        priority = int(digest([seed, r['id']]), 16)
        entry = (-priority, r['id'], r)
        heap = buckets[key]
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    result = []
    for stage in ('knowdist', 'icldist'):
        train = []
        for (s, split, _), entries in sorted(buckets.items()):
            if s != stage:
                continue
            ordered = [v[2] for v in sorted(entries, key=lambda x: (-x[0], x[1]))]
            if split == 'train':
                train.append(ordered)
            else:
                result.extend(ordered)
        count = 0
        for index in range(max(map(len, train), default=0)):
            for group in train:
                if index < len(group) and count < caps[stage]:
                    result.append(group[index])
                    count += 1
    return result


def select_rows(rows, method, token_budget, max_records, seed, scores=None, min_ifd=0., max_ifd=1.):
    """B is coverage-balanced; A ranks IFD within each group. No test feedback.

    Bounds are explicit experimental controls, NOT verified quality thresholds.
    Shared eligibility/candidate pool, strict no-overshoot selection budget.
    """
    if method not in ('random', 'coverage', 'student') or token_budget < 1 or max_records < 1:
        raise ValueError('invalid selection controls')
    if not 0 <= min_ifd < max_ifd or not math.isfinite(max_ifd):
        raise ValueError('invalid IFD interval')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate candidate IDs')
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('selection must use train only')
    buckets = defaultdict(list)
    for r in sorted(rows, key=lambda r: r['id']):
        if len(r['input_ids']) < 2 or not any(t != -100 for t in r['labels'][1:]):
            raise ValueError('invalid tokenized record')
        if method == 'student':
            if scores is None or r['id'] not in scores:
                raise ValueError(f'missing score for {r["id"]}; no silent fallback')
            score = scores[r['id']]
            if not math.isfinite(score):
                raise ValueError('non-finite IFD')
            if not min_ifd <= score <= max_ifd:
                continue
        buckets[r['group'] if method != 'random' else '__all__'].append(r)
    for group, values in buckets.items():
        random.Random(f'{seed}:{group}').shuffle(values)
        if method == 'student':
            values.sort(key=lambda r: -scores[r['id']])
    selected, tokens, group_tokens = [], 0, Counter()
    cursors = Counter()
    while buckets and len(selected) < max_records:
        # Token coverage, rather than equal record count, avoids long groups dominating.
        group = min(buckets, key=lambda g: (group_tokens[g], g))
        values = buckets[group]
        chosen = None
        while cursors[group] < len(values):
            r = values[cursors[group]]
            cursors[group] += 1
            if tokens + len(r['input_ids']) <= token_budget:
                chosen = r
                break
        if chosen is None:
            del buckets[group]
            continue
        selected.append(chosen)
        size = len(chosen['input_ids'])
        tokens += size
        group_tokens[group] += size
    if not selected:
        raise ValueError('empty selection; inspect score interval/budget/candidate pool')
    return selected, {'method': method, 'tokens': tokens, 'token_budget': token_budget,
                      'unused_tokens': token_budget - tokens, 'records': len(selected),
                      'groups': dict(Counter(r['group'] for r in selected)),
                      'group_tokens': dict(group_tokens),
                      'missing_groups': sorted({r['group'] for r in rows} - {r['group'] for r in selected})}


def steps_for_tokens(rows, target, batch_size, accumulation, seed):
    """Simulate trainer's exact shuffled stream up to an optimizer boundary."""
    if not rows or min(target, batch_size, accumulation) < 1:
        raise ValueError('invalid training budget')
    epoch, offset, steps, tokens, order = 0, 0, 0, 0, []
    sizes = [len(r['input_ids']) for r in rows]
    while tokens < target:
        for _ in range(batch_size * accumulation):
            if not order:
                order = list(range(len(rows)))
                random.Random(seed + epoch).shuffle(order)
            tokens += sizes[order[offset]]
            offset += 1
            if offset == len(order):
                epoch, offset, order = epoch + 1, 0, []
        steps += 1
    return steps, tokens
