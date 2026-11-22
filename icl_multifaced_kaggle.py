from notebook_runtime import (add_runtime_arguments, prepare_runtime, load_model,
    select_examples, configure_tokenizer, limit_prompts, run_metadata, PredictionCache,
    save_result_kaggle as save_result)
from tqdm import tqdm
import argparse
from utils import save_line_json, load_line_json
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import load_json,save_json
from sklearn.metrics import f1_score, accuracy_score, classification_report
import datetime
import time
import os


id_to_label = {
    'multifaced/tweeteval': [' anger', ' joy', ' sadness', ' optimism'],
    'multifaced/intimacy': [' not intimate', ' moderately intimate', ' highly intimate'],
    'multifaced/offenseval':[' non-offensive', ' offensive'],
    'multifaced/irony18': [' non-irony', ' irony'],
    'multifaced/pstance': [' against', ' favor'],
    'sc/imdb': [' negative', ' positive'],
    'sc/sst2': [' negative', ' positive'],
    'sc/yelp2': [' negative', ' positive'],
    'sc/twitter': [' negative', ' positive', ' neutral'],
}


def get_label_list(tokenizer, dataset):
    ids = [tokenizer(label, add_special_tokens=False)['input_ids'][0]
           for label in id_to_label[dataset]]
    if len(set(ids)) != len(ids):
        raise ValueError('Label first tokens collide for this tokenizer; use a compatible model or implement full-label scoring.')
    return ids


def cal_metric(results, y_true, y_pred, seed, subname, model_name_or_path, dataset, output_dir, k_shot, metadata=None):
    model_name = model_name_or_path.split('/')[-1]
    print(model_name)
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f'{subname}_{model_name}_{dataset}_{seed}_numOfshot{k_shot}_{current_time}'
    output_path = os.path.join(output_dir, file_name + '.json')
    print(output_path)

    print(Counter(y_true))
    print(Counter(y_pred))

    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print(f'f1: {f1:.4f} | accuracy: {accuracy:.4f}\n')


    save_result(output_dir, file_name, f1, accuracy, metadata)
    save_json(results, output_path)


class DataCollator:
    def __init__(self, tokenizer, max_seq_length, mode):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

    def tok(self, text, max_seq_length):
        kwargs = {
            'text': text,
            'return_tensors': 'pt',
        }

        if max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True

        else:
            kwargs['max_length'] = max_seq_length
            kwargs['padding'] = True
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        return batch_encodings

    def __call__(self, examples):

        texts = [example['prompts'] for example in examples]
        batch_encodings = self.tok(texts, self.max_seq_length)

        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'examples': examples
        }

def get_dataloader(test_examples, mode, batch_size, shuffle, tokenizer):
    dataloader = DataLoader(
        dataset=test_examples,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
        collate_fn=DataCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            mode=mode,
        )
    )

    print('dataloader-' + mode, len(dataloader))
    return dataloader


def test_dataloader(test_examples, eval_batch_size,tokenizer):
    return get_dataloader(test_examples, "test", eval_batch_size, shuffle=False, tokenizer=tokenizer)


def construct_prompt(prompt, test_examples, demo_pool, k_shot):

    for example in test_examples:
        text = example['sentence']
        demos = random.sample(demo_pool, k=k_shot)
        prompts = [prompt]
        prompts.append('')
        for demo in demos:
            prompts.append(f"Sentence: {demo['sentence']}\nLabel: {demo['label'].lower()}\n")
        prompts.append(f"Sentence: {example['sentence']}\nLabel:")
        example['prompts'] = '\n'.join(prompts)



    return test_examples


def construct_prompt_pstance(prompt, test_examples, demo_pool, k_shot):

    for example in test_examples:
        text = example['sentence']
        target = example['target']
        demos = random.sample(demo_pool, k=k_shot)
        prompts = [prompt % target]
        prompts.append('')
        for demo in demos:
            prompts.append(f"Sentence: {demo['sentence']}\nLabel: {demo['label'].lower()}(opinion towards '{demo['target']}')\n")
        prompts.append(f"Sentence: {example['sentence']}\nLabel:")
        example['prompts'] = '\n'.join(prompts)



    return test_examples



def load_dataset(data_path, demo_split='train', eval_split='test'):
    test_file_name = os.path.join(data_path, eval_split + '.json')
    test_examples = load_json(test_file_name)

    demo_pool_file_name = os.path.join(data_path, demo_split + '.json')
    demo_examples = load_json(demo_pool_file_name)

    raw_datasets = {
        'test': test_examples,
        'demo': demo_examples
    }

    print('-----------data statistic-------------')
    for mode in ('test','demo'):
        num_samples = len(raw_datasets[mode])
        print(f'{mode.upper():<5} | Samples: {num_samples:<5}')
    print('--------------------------------------')



    return raw_datasets



import sys
import argparse
import datetime
if __name__ == '__main__':
    print('hello')
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, )
    parser.add_argument('--subname', required=True, )
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--model_name_or_path', default=None )
    parser.add_argument('--model_version', default=None)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--data_prop', type=float, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(__file__), 'datasets'))
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--k_shot', type=int, default=4)

    add_runtime_arguments(parser)
    args = parser.parse_args()
    prepare_runtime(args)

    dataset = args.dataset
    subname = args.subname
    seed = args.seed
    model_version = args.model_version
    model_name_or_path = args.model_name_or_path
    max_seq_length = args.max_seq_length
    data_prop = args.data_prop
    eval_batch_size = args.eval_batch_size
    data_dir = args.data_dir
    output_dir = args.output_dir
    k_shot = args.k_shot

    data_path = os.path.join(data_dir, dataset)
    torch.manual_seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)


    model_name_or_path = args.model_name_or_path

    print('-'*20+'Experiment Setup'+'-'*20)
    print('data_path: ', data_path)
    print(f' subname: {subname}\n dataset: {dataset}\n seed:{seed}\n model_name_or_path:{model_name_or_path}\n data_path:{data_path}\n output_dir:{output_dir}\n eval_batch_size:{eval_batch_size}\n data_prop:{data_prop}\n max_seq_length:{max_seq_length}\n numbers_of_shot:{k_shot}')
    print('-' * 50)

    prompts = {
        'multifaced/irony18':
            """Please perform Irony Detection task. Given the sentence, assign a sentiment label from ['irony', 'non-irony']. Return label only without any other text.""",
        'multifaced/offenseval':
            """Please perform Offensive Detection task. Given the sentence, assign a sentiment label from ['non-offensive', 'offensive']. Return label only without any other text.""",
        'multifaced/intimacy':
            """Please perform Intimacy Detection task. Given the sentence, assign an intimacy label from ['not intimate', 'moderately intimate', 'highly intimate']. Return label only without any other text.""",
        'multifaced/tweeteval':
            """Please perform Emotion Detection task. Given the sentence, assign a emotion label from ['anger', 'joy', 'sadness', 'optimism']. Return the label only without any other text.""",
        'multifaced/pstance':
            """Please perform Stance Detection task. Given the sentence, assign a sentiment label expressed by the author towards "%s" from ['against', 'favor']. Return label only without any other text.""",
        'sc/imdb':
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive']. Return label only without any other text.""",
        'sc/sst2':
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive']. Return label only without any other text.""",
        'sc/yelp2':
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive']. Return label only without any other text.""",
        'sc/twitter':
            """Please perform Sentiment Analysis task. Given the sentence, assign a sentiment polarity label from ['negative', 'positive', 'neutral']. Return label only without any other text.""",
    }




    raw_datasets = select_examples(load_dataset(data_path, args.demo_split, args.eval_split), args)

    if dataset == 'multifaced/pstance':
        test_examples = construct_prompt_pstance(prompts[dataset], raw_datasets['test'], raw_datasets['demo'], k_shot)
    else:
        test_examples = construct_prompt(prompts[dataset], raw_datasets['test'], raw_datasets['demo'], k_shot)

    for i in range(1):
        print(test_examples[i]['prompts'])
        print('-'*100)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=args.model_revision, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token



    model = load_model(args)
    configure_tokenizer(tokenizer, model, args, 1)
    max_seq_length = args.max_seq_length
    truncated = limit_prompts(test_examples, tokenizer, max_seq_length)
    metadata = run_metadata(args, raw_datasets, truncated)
    cache = PredictionCache(args.resume_file, metadata)
    pending = [example for example in test_examples if cache.get(example) is None]
    dataloader = test_dataloader(pending, eval_batch_size, tokenizer)





    model.eval()

    label_ids = get_label_list(tokenizer, dataset)
    results = []
    timings, measured_examples, warmed = [], 0, False
    cuda = torch.cuda.is_available()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            if args.measure_inference and not warmed:
                warmup = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                del warmup
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                warmed = True
            if args.measure_inference:
                if cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
            # Same first-token decision rule as the original evaluator; no KV cache.
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            if args.measure_inference:
                if cuda:
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                measured_examples += len(batch['examples'])
            scores = output.logits[:, -1, label_ids].float().cpu()
            for example, score in zip(batch['examples'], scores):
                cache.put(example, {'prediction': id_to_label[dataset][score.argmax().item()].strip(),
                                    'prediction_score': score.tolist()})
            cache.flush()
            del output, scores, input_ids, attention_mask

    for example in test_examples:
        result = dict(example)
        result.update(cache.get(example))
        results.append(result)
    labels = [example['label'].lower() for example in results]
    predictions = [example['prediction'] for example in results]
    result_metadata = dict(metadata)
    if args.measure_inference:
        total = sum(timings)
        ordered = sorted(timings)
        import math
        result_metadata['inference_measurement'] = {
            'scope': 'Model forward only; excludes loading, CPU tokenization, host-to-device copies and scoring; one warm-up batch.',
            'measured_examples': measured_examples, 'measured_batches': len(timings),
            'batch_size': args.eval_batch_size, 'forward_seconds': total,
            'examples_per_second': measured_examples / total if total else None,
            'amortized_ms_per_example': total * 1000 / measured_examples if measured_examples else None,
            'mean_batch_latency_ms': total * 1000 / len(timings) if timings else None,
            'p95_batch_latency_ms': ordered[math.ceil(.95 * len(ordered)) - 1] * 1000 if ordered else None,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if cuda and warmed else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if cuda and warmed else None,
        }
    cal_metric(results, labels, predictions, seed, subname, model_name_or_path,
               dataset, output_dir, k_shot, result_metadata)
