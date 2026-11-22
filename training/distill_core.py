"""Dependency-free data checks, splits and adaptive group weights."""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_records(path):
    with open(path, encoding='utf-8-sig') as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                obj = json.loads(text)
                if not isinstance(obj, dict):
                    raise ValueError('record is not an object')
                yield line, obj
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid JSON record at line {line}: {exc}') from exc


def mapped_record(obj, prompt_fields, response_field, group_field, source_field):
    """Explicit field order; preserve each field's contents, including whitespace."""
    parts = []
    for field in prompt_fields:
        value = obj.get(field)
        if value is None:
            raise ValueError(f'missing prompt field: {field}')
        if not isinstance(value, str):
            raise ValueError(f'prompt field must be a string: {field}')
        if value.strip():
            parts.append(value)
    response = obj.get(response_field)
    if not parts or not isinstance(response, str) or not response.strip():
        raise ValueError('empty prompt or response')
    group = obj.get(group_field)
    if group is None or not str(group).strip():
        raise ValueError(f'missing group field: {group_field}')
    source = obj.get(source_field)
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f'missing source/input field for group split: {source_field}')
    # Normalize only the split key; never alter the training target.
    source_key = ' '.join(source.split()).casefold()
    return {'prompt': '\n\n'.join(parts), 'response': response,
            'group': str(group), 'source_key': digest(source_key)}


def split_records(records, seed, dev_fraction=0.1, max_train=0, probe_per_group=4,
                  holdout_per_group=4, min_group_train=8, max_groups=32):
    if not 0 < dev_fraction < 1:
        raise ValueError('dev_fraction must be between 0 and 1')
    if min(probe_per_group, holdout_per_group, min_group_train, max_groups) < 1 or max_train < 0:
        raise ValueError('Invalid split limits')
    # Assign every input group before selecting a pilot subset. No input crosses splits.
    keys = sorted({r['source_key'] for r in records})
    random.Random(seed).shuffle(keys)
    ndev = max(2, round(len(keys) * dev_fraction))
    if ndev >= len(keys):
        raise ValueError('Not enough independent inputs to split')
    dev_keys = set(keys[:ndev])
    probe_keys = set(keys[:ndev // 2])
    train = [r for r in records if r['source_key'] not in dev_keys]
    random.Random(seed).shuffle(train)
    if max_train:
        train = train[:max_train]
    counts = Counter(r['group'] for r in train)
    # Group collapsing uses TRAIN only. Reserve one slot for rare groups.
    common = [g for g, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
              if n >= min_group_train][:max_groups - 1]
    mapping = {g: g if g in common else '__rare__' for g in counts}
    groups = sorted(set(mapping.values()))
    for r in train:
        r['train_group'] = mapping[r['group']]
    probe, holdout = [], []
    for allowed, dest, limit in ((probe_keys, probe, probe_per_group),
                                  (dev_keys - probe_keys, holdout, holdout_per_group)):
        buckets = defaultdict(list)
        for r in records:
            if r['source_key'] in allowed and r['group'] in mapping:
                r['train_group'] = mapping[r['group']]
                buckets[r['train_group']].append(r)
        for g in groups:
            items = sorted(buckets[g], key=lambda r: r['id'])
            random.Random(f'{seed}:{g}').shuffle(items)
            dest.extend(items[:limit])
    if not train or not probe or not holdout:
        raise ValueError('Empty train/probe/holdout; increase pilot size or dev fraction')
    return train, probe, holdout, mapping


class GroupWeights:
    """Progress normalized by initial per-response CE; bounded mean-one weights."""
    def __init__(self, groups, counts, mode='progress', mix=0.5, temperature=0.05,
                 ema=0.5, floor=0.5, ceiling=2.0):
        if mode not in ('uniform', 'balanced', 'loss', 'progress'):
            raise ValueError('Unknown weighting mode')
        if not 0 <= mix <= 1 or temperature <= 0 or not 0 <= ema < 1:
            raise ValueError('Invalid mix/temperature/EMA')
        if not 0 < floor <= 1 <= ceiling:
            raise ValueError('Require 0 < floor <= 1 <= ceiling')
        self.groups = list(groups)
        self.counts = dict(counts)
        self.mode, self.mix, self.temperature, self.ema = mode, mix, temperature, ema
        self.floor, self.ceiling = floor, ceiling
        self.initial, self.previous, self.smoothed = {}, {}, {}
        self.weights = {g: 1.0 for g in groups}
        if mode == 'balanced':
            self.weights = self._bounded({g: 1 / counts[g] for g in groups})

    def _bounded(self, raw):
        # Solve E_train[clip(scale * raw, floor, ceiling)] = 1.
        total = sum(self.counts.values())
        lo, hi = 0., 1.
        def values(scale):
            return {g: min(self.ceiling, max(self.floor, scale * raw[g])) for g in self.groups}
        def mean(w):
            return sum(w[g] * self.counts[g] for g in self.groups) / total
        while mean(values(hi)) < 1 - 1e-12:
            hi *= 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if mean(values(mid)) < 1:
                lo = mid
            else:
                hi = mid
        return values((lo + hi) / 2)

    def update(self, losses):
        if not losses or any(not math.isfinite(v) or v < 0 for v in losses.values()):
            raise ValueError('Probe losses must be nonnegative and finite')
        scores = {}
        for g, value in losses.items():
            if g not in self.weights:
                raise ValueError(f'Unknown probe group: {g}')
            if g not in self.initial:
                self.initial[g] = self.previous[g] = self.smoothed[g] = value
                scores[g] = 0.
                continue
            smooth = self.ema * self.smoothed[g] + (1 - self.ema) * value
            progress = max(0., (self.smoothed[g] - smooth) / max(self.initial[g], 1e-8))
            scores[g] = progress if self.mode == 'progress' else smooth / max(self.initial[g], 1e-8)
            self.previous[g], self.smoothed[g] = self.smoothed[g], smooth
        if self.mode in ('uniform', 'balanced'):
            return self.weights
        # Groups without probes retain a neutral score; coverage is reported.
        scores = {g: scores.get(g, 0.) for g in self.groups}
        peak = max(scores.values())
        exp = {g: math.exp((v - peak) / self.temperature) for g, v in scores.items()}
        avg = sum(exp.values()) / len(exp)
        raw = {g: (1 - self.mix) + self.mix * exp[g] / avg for g in self.groups}
        self.weights = self._bounded(raw)
        return self.weights


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)
