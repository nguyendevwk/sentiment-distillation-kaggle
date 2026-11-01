"""Summarize modern result metadata and legacy SentiBench result names."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

DATASETS = ['sc/imdb', 'sc/yelp2', 'sc/sst2', 'sc/twitter',
            'multifaced/irony18', 'multifaced/tweeteval', 'multifaced/pstance',
            'multifaced/intimacy', 'absa/asqp_rest16', 'absa/opener',
            'absa/atsa_rest16', 'absa/acsa_rest16']


def summarize(path, experiments=None):
    groups = defaultdict(lambda: defaultdict(dict))
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        name, metrics = json.loads(line)
        metadata = metrics.get('metadata')
        if metadata:
            experiment = metadata['experiment']
            dataset = metadata['dataset']
            settings = json.dumps(metadata['settings'], sort_keys=True)
            seed = metadata['seed']
        else:
            match = re.fullmatch(r'(.+)_((?:sc|multifaced|absa)/.+)_(-?\d+)_numOfshot(\d+)_(\d{4}(?:_\d{2}){5})', name)
            if not match:
                raise ValueError(f'Unrecognized legacy result name: {name}')
            experiment, dataset, seed, shots, _ = match.groups()
            settings = json.dumps({'legacy': True, 'k_shot': int(shots)}, sort_keys=True)
        if experiments and not any(e in experiment for e in experiments):
            continue
        # Re-running the same seed replaces its older result.
        groups[(experiment, settings)][dataset][str(seed)] = metrics['f1']
    if not groups:
        print('No completed evaluation results found.')
    for (experiment, settings), datasets in groups.items():
        print(f'\nExperiment: {experiment}\nSettings: {settings}')
        means = []
        for dataset in DATASETS:
            values = list(datasets.get(dataset, {}).values())
            if values:
                mean = sum(values) / len(values) * 100
                means.append(mean)
                print(f'{dataset:<26} {mean:7.2f}  ({len(values)} seed(s))')
            else:
                print(f'{dataset:<26}     N/A')
        if len(means) == len(DATASETS):
            print(f'SentiBench average: {sum(means) / len(means):.2f}')
        elif means:
            print(f'Partial average ({len(means)}/12 datasets): {sum(means) / len(means):.2f}; not the full benchmark score')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./output/result.txt')
    parser.add_argument('--experiments', nargs='*')
    args = parser.parse_args()
    summarize(args.path, args.experiments)
