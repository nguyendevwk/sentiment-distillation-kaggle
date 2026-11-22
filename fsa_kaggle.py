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

from parse_utils.asqp import Result_asqp
from parse_utils.ssa import Result_ssa
from parse_utils.coqe import Result_coqe
from parse_utils.atsa import Result_atsa
from parse_utils.acsa import Result_acsa

from parse_utils import asqp as _asqp, ssa as _ssa, atsa as _atsa, acsa as _acsa
from parse_utils.structured_literal import asqp_prediction, ssa_prediction, atsa_prediction, acsa_prediction
_asqp.parse_quards_for_predict = asqp_prediction
_ssa.parse_quards_for_predict = ssa_prediction
_atsa.parse_aspect_polarity = atsa_prediction
_acsa.parse_aspect_polarity = acsa_prediction


def ifnt0(x):
    """
    Return 0 if the input is None, otherwise return the input.
    :param x: The input value that may be None.
    :return: 0 if x is None, x otherwise.
    """
    return 0 if x is None else x

class FineTuneDataset(Dataset):
    def __init__(self, dataset_name, dataset, demo_pool ,k_shot, model_name_or_path, mode, max_length, model_revision=None):
        self.model_name_or_path = model_name_or_path
        self.demo_pool = demo_pool
        self.k_shot = k_shot
        self.dataset_name = dataset_name
        self.dataset_dict = {dataset_name: dataset}
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, revision=model_revision, use_fast=True, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.mode = mode
        self.prompts = {
        'absa/asqp_rest16':"""Please perform Aspect Sentiment Quad Prediction task. Given the sentence, extract all (aspect category, aspect term, opinion, sentiment polarity) quadruples.

1. Aspect category should be selected from ['ambience general', 'drinks prices','drinks quality', 'drinks style_options', 'food general', 'food prices', 'food quality', 'food style_options', 'location general', 'restaurant general', 'restaurant miscellaneous', 'restaurant prices', 'service general'].

2. Sentiment polarity should be selected from ['negative', 'neutral', 'positive'].

3. If there is no aspect term, use 'NULL' as the aspect term. Only aspect term can be 'NULL', aspect category, opinion and sentiment polarity CANNOT be 'NULL'.

4. Please return python list only, without any other comments or texts.""",
        'absa/asqp_laptop':"""Please perform Aspect Sentiment Quad Prediction task. Given the sentence, extract all (aspect category, aspect term, opinion, sentiment polarity) quadruples.

1. Aspect category should be selected from ['battery miscellaneous', 'battery operation_performance', 'battery quality', 'company general', 'cpu design_features', 'cpu miscellaneous', 'cpu operation_performance', 'cpu quality', 'display design_features', 'display general', 'display operation_performance', 'display quality', 'display usability', 'fans_cooling design_features', 'fans_cooling operation_performance', 'fans_cooling quality', 'graphics design_features', 'graphics general', 'graphics miscellaneous', 'graphics quality', 'hard_disc design_features', 'hard_disc quality', 'hardware general', 'hardware operation_performance', 'hardware quality', 'keyboard design_features', 'keyboard general', 'keyboard operation_performance', 'keyboard quality', 'keyboard usability', 'laptop connectivity', 'laptop design_features', 'laptop general', 'laptop miscellaneous', 'laptop operation_performance', 'laptop portability', 'laptop price', 'laptop quality', 'laptop usability', 'memory design_features', 'motherboard quality', 'mouse design_features', 'mouse general', 'mouse operation_performance', 'mouse quality', 'mouse usability', 'multimedia_devices design_features', 'multimedia_devices general', 'multimedia_devices miscellaneous', 'multimedia_devices operation_performance', 'multimedia_devices quality', 'multimedia_devices usability', 'optical_drives operation_performance', 'optical_drives quality', 'os design_features', 'os general', 'os miscellaneous', 'os operation_performance', 'os quality', 'os usability', 'ports design_features', 'ports operation_performance', 'ports quality', 'power_supply design_features', 'power_supply miscellaneous', 'power_supply operation_performance', 'power_supply quality', 'shipping price', 'shipping quality', 'software design_features', 'software general', 'software miscellaneous', 'software operation_performance', 'software price', 'software quality', 'software usability', 'support miscellaneous', 'support price', 'support quality', 'warranty general', 'warranty price'].

2. Sentiment polarity should be selected from ['negative', 'neutral', 'positive'].

3. If there is no aspect term, use 'NULL' as the aspect term. Only aspect term can be 'NULL', aspect category, opinion and sentiment polarity CANNOT be 'NULL'.

4. Please return python list only, without any other comments or texts.""",

            'absa/opener':"""Please perform the Structured Sentiment Analysis task. Given a sentence, extract all opinion tuples in the format (holder, target, sentiment expression, sentiment polarity). 

Each tuple should contain:

- Holder: The entity expressing the sentiment, if there is no explicit holder, use 'NULL' as the holder.

- Target: The entity being evaluated, if there is no explicit target, use 'NULL' as the target.

- Sentiment Expression: The phrase conveying the sentiment.

- Sentiment Polarity: The polarity of the sentiment, either positive, negative, or neutral.

Follow these rules:

1. If there is no sentiment expression, return 'NULL' for all fields.

2. Please return python list only, without any other comments or texts.""",

            'absa/dsunis':"""Please perform the Structured Sentiment Analysis task. Given a sentence, extract all opinion tuples in the format (holder, target, sentiment expression, sentiment polarity). 

Each tuple should contain:

- Holder: The entity expressing the sentiment, if there is no explicit holder, use 'NULL' as the holder.

- Target: The entity being evaluated, if there is no explicit target, use 'NULL' as the target.

- Sentiment Expression: The phrase conveying the sentiment.

- Sentiment Polarity: The polarity of the sentiment, either positive, negative, or neutral.

Follow these rules:

1. If there is no sentiment expression, return 'NULL' for all fields.

2. Please return python list only, without any other comments or texts.""",
            
            'absa/coqe_camera':"""Please perform the Comparative Opinion Quintuple Extraction task. Given a sentence, extract all opinion tuples in the format (subject, object, comparative aspect, comparative opinion, comparative preference).

Each tuple should contain:

- subject: The first entity being compared.

- object: The second entity being compared.

- comparative aspect: The feature or attribute being compared.

- comparative opinion: The opinion expression indicating the comparative preference between two entities.

- comparative preference: One of the four possible comparative relationships:

    Worse: The subject is worse than the object.

    Equal: The subject is equal to the object.

    Better: The subject is better than the object.

    Different: The subject is different from the object.

If any component is missing, use NULL as its placeholder. Please extract all opinion tuples in the format (subject, object, comparative aspect, comparative opinion, comparative preference). Please return python list only, without any other comments or texts.""",
    "absa/atsa_rest16":"Please perform Aspect Term Sentiment Analysis task. Given the sentence, extract all (aspect term, sentiment polarity) pairs.",
    "absa/acsa_rest16":"Please perform aspect-level sentiment analysis task. Given the sentence, tag all (aspect category, sentiment) pairs. Aspect category should be selected from ['ambience general', 'drinks prices', 'drinks quality', 'drinks style_options', 'food general', 'food prices', 'food quality', 'food style_options', 'location general', 'restaurant general', 'restaurant miscellaneous', 'restaurant prices', 'service general'], and sentiment should be selected from ['negative', 'neutral', 'positive']. If there are no target-sentiment pairs, return an empty list. Otherwise return a python list of tuples containing two strings in double quotes. Please return python list only, without any other comments or texts.",
        }
        self.construct_sample()

    def __len__(self):
        return len(self.dataset)


    def add_prompt(self, dataset_name, data):

        prompt = self.prompts[dataset_name]
        prompts = [prompt]
        prompts.append('')


        sentence = data['sentence']
        demos = random.sample(self.demo_pool, k=self.k_shot)
        data['demo_ids'] = [demo.get('_demo_id') for demo in demos]

        for demo in demos:
            prompts.append(f"Sentence: {demo['sentence']}\nLabel: {demo['label_seq']}\n")

        
        prompts.append(f"Sentence: {sentence}\nLabel:")
        data['prompts'] = '\n'.join(prompts)
        data['dataset'] = dataset_name


    def make_label_seq(self, dataset_name, data):

        def make_atsa_rest16_seq(example):
            if 'label_seq' in example:
                return example['label_seq']

            atsa_seq = []
            for aspect in example['aspects']:
                polarity = aspect['polarity']
                aspect_term = aspect['target']

                if aspect_term in ('NULL', None):
                    continue

                assert aspect_term in example['sentence']

                loc = (ifnt0(aspect['from']), ifnt0(aspect['to']))

                atsa_seq.append((aspect_term, polarity, loc))

            atsa_seq = list(set(atsa_seq))
            atsa_seq = sorted(atsa_seq, key=lambda it: it[-1])
            atsa_seq = [(aspect_term, polarity) for aspect_term, polarity, _ in atsa_seq]
            example['label_seq'] = atsa_seq


        def make_acsa_rest16_seq(example):
            if 'label_seq' in example:
                return example['label_seq']


            acsa_seq = []
            for aspect in example['aspects']:
                polarity = aspect['polarity']
                category = aspect['category'].lower().replace('#', ' ')

                category_polarity = (category, polarity)
                if category_polarity not in acsa_seq:
                    acsa_seq.append(category_polarity)

            example['label_seq'] = acsa_seq


        def make_asqp_seq(example):
            if 'label_seq' in example:
                print(1)
                return example['label_seq']


            asqp_seq = []
            for quard in example['quards']:

                aspect = quard['aspect']
                category = quard['category']
                sentiment = quard['sentiment']
                opinion = quard['opinion']



                asqp_quard = (category, aspect, opinion, sentiment)
                if asqp_quard not in asqp_seq:
                    asqp_seq.append(asqp_quard)

            example['label_seq'] = asqp_seq



        def make_ssa_seq(example):
            if 'label_seq' in example:
                return example['label_seq']

            ssa_seq = []
            for opinion in example['opinions']:

                if len(opinion['Source'][0]) == 0:
                    holder = 'NULL'
                else:
                    holder = ' AND '.join(opinion['Source'][0])

                if len(opinion['Target'][0]) == 0:
                    target = 'NULL'
                else:
                    target = ' AND '.join(opinion['Target'][0])

                expressions = opinion['Polar_expression'][0]
                expressions = ' AND '.join(expressions)

                sentiment = opinion['Polarity'].lower()

                sst_quard = (holder, target, expressions, sentiment)
                if sst_quard not in ssa_seq:
                    ssa_seq.append(sst_quard)

            if example['opinions'] == []:
                ssa_seq.append(('NULL', 'NULL', 'NULL', 'NULL'))

            example['label_seq'] = ssa_seq

        def make_multifaced_seq(example):
            if 'label_seq' in example:
                return example['label_seq']
            example['label_seq'] = example['label'].lower()

        def make_coqe_seq(example):
            if 'label_seq' in example:
                return example['label_seq']


            coqe_seq = []
            for quintuple in example['quintuple']:

                subject = quintuple['subject'] if quintuple['subject'] != '' else 'NULL'
                object = quintuple['object'] if quintuple['object'] != '' else 'NULL'
                aspect = quintuple['comparative aspect'] if quintuple['comparative aspect'] != '' else 'NULL'
                opinion = quintuple['comparative opinion'] if quintuple['comparative opinion'] != '' else 'NULL'
                preference = quintuple['comparative preference'] if quintuple['comparative preference'] != '' else 'NULL'


                coqe_quard = (subject, object, aspect, opinion, preference)
                if coqe_quard not in coqe_seq:
                    coqe_seq.append(coqe_quard)

            example['label_seq'] = coqe_seq




        if dataset_name == 'absa/asqp_rest16' or dataset_name == 'absa/asqp_laptop':
            make_asqp_seq(data)
        elif dataset_name == 'absa/dsunis' or dataset_name == 'absa/opener':
            make_ssa_seq(data)
        elif dataset_name == 'absa/coqe_camera':
            make_coqe_seq(data)
        elif dataset_name == 'absa/atsa_rest16':
            make_atsa_rest16_seq(data)
        elif dataset_name == 'absa/acsa_rest16':
            make_acsa_rest16_seq(data)
        else:
            make_multifaced_seq(data)


    def construct_sample(self):


        print(self.dataset_name)
        for item in self.demo_pool:
            self.make_label_seq(self.dataset_name, item)


        for dataset_name, dataset in self.dataset_dict.items():
            for data in dataset:
                self.add_prompt(dataset_name, data)
                self.make_label_seq(dataset_name, data)


        self.dataset = []
        for dataset in self.dataset_dict.values():
            self.dataset.extend(dataset)
        self.dataset = self.dataset



    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item['prompts']
        label_seq = item['label_seq']

        return item



def cal_metric(results, f1, seed, subname, model_name_or_path, dataset, output_dir, k_shot, metadata=None):
    model_name = model_name_or_path.split('/')[-1]
    print(model_name)
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f'{subname}_{model_name}_{dataset}_{seed}_numOfshot{k_shot}_{current_time}'
    output_path = os.path.join(output_dir, file_name + '.json')
    print(output_path)




    save_result(output_dir, file_name, f1, None, metadata)
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


# 从test.json 和 train.json 中加载数据
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
    print('model name: ', model_name_or_path)
    print('-'*20+'Experiment Setup'+'-'*20)
    print('data_path: ', data_path)
    print(f' subname: {subname}\n dataset: {dataset}\n seed:{seed}\n model_name_or_path:{model_name_or_path}\n data_path:{data_path}\n output_dir:{output_dir}\n eval_batch_size:{eval_batch_size}\n data_prop:{data_prop}\n max_seq_length:{max_seq_length}\n numbers_of_shot:{k_shot}')
    print('-' * 50)


    raw_datasets = select_examples(load_dataset(data_path, args.demo_split, args.eval_split), args)



    test_examples = FineTuneDataset(dataset, raw_datasets['test'], raw_datasets['demo'], k_shot, model_name_or_path, 'test', max_seq_length, model_revision=args.model_revision)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=args.model_revision, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token


    
    model = load_model(args)
    configure_tokenizer(tokenizer, model, args, args.max_new_tokens)
    max_seq_length = args.max_seq_length
    truncated = limit_prompts(test_examples, tokenizer, max_seq_length)
    metadata = run_metadata(args, raw_datasets, truncated)
    cache = PredictionCache(args.resume_file, metadata)
    pending = [example for example in test_examples if cache.get(example) is None]
    dataloader = test_dataloader(pending, eval_batch_size, tokenizer)


    print(model_name_or_path)
    model.eval()

    eval_start, warmed = time.monotonic(), False
    cuda = torch.cuda.is_available()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            options = dict(attention_mask=attention_mask, max_new_tokens=args.max_new_tokens,
                           num_beams=1, num_return_sequences=1, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
            if args.measure_inference and not warmed:
                warmup = model.generate(input_ids, **{**options, 'max_new_tokens': min(8, args.max_new_tokens)})
                del warmup
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                warmed = True
            if args.measure_inference:
                if cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
            generated_ids = model.generate(input_ids, **options)
            if args.measure_inference:
                if cuda:
                    torch.cuda.synchronize()
                duration = time.perf_counter() - started
            tails = generated_ids[:, input_ids.shape[1]:]
            generateds = tokenizer.batch_decode(tails, skip_special_tokens=True)
            if args.measure_inference:
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos or [])
                output_tokens = 0
                for tail in tails.tolist():
                    output_tokens += next((i + 1 for i, token in enumerate(tail) if token in eos), len(tail))
                cache.measure(duration, len(batch['examples']), output_tokens,
                    allocated=torch.cuda.max_memory_allocated() if cuda else 0,
                    reserved=torch.cuda.max_memory_reserved() if cuda else 0)
            for example, prediction in zip(batch['examples'], generateds):
                cache.put(example, prediction.strip())
            cache.flush()
            del generated_ids, tails, input_ids, attention_mask
            if args.max_eval_seconds and time.monotonic() - eval_start >= args.max_eval_seconds:
                break

    if any(cache.get(example) is None for example in test_examples):
        print('Evaluation paused; prediction cache saved. No partial F1 reported.', flush=True)
        raise SystemExit(0)
    result_metadata = dict(metadata)
    if args.measure_inference:
        result_metadata['inference_measurement'] = cache.measurement_summary('Autoregressive generation', args.eval_batch_size)

    outputs = [(example, cache.get(example)) for example in test_examples]

    current_results = []

    if  dataset == 'absa/asqp_rest16' or dataset == 'absa/asqp_laptop': 
        current_results.append(Result_asqp.parse_from(outputs))
    elif dataset == 'absa/opener' or dataset == 'absa/dsunis':
        current_results.append(Result_ssa.parse_from(outputs))
    elif dataset == 'absa/coqe_camera':
        current_results.append(Result_coqe.parse_from(outputs))
    elif dataset == 'absa/atsa_rest16':
        current_results.append(Result_atsa.parse_from(outputs))
    elif dataset == 'absa/acsa_rest16':
        current_results.append(Result_acsa.parse_from(outputs))
    else:
        raise ValueError(f'No dataset named {dataset}')


    for result in current_results:
        result.cal_metric()




    current_results[0].report()
    f1_scores = current_results[0].monitor
    print(f1_scores)
    
    
    
    cal_metric(outputs,f1_scores, seed, subname, model_name_or_path, dataset, output_dir, k_shot, result_metadata)
