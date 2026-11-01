# Targeted Distillation for Sentiment Analysis

## Kaggle / Colab notebook

Upload [Sentiment_Distillation_Kaggle_Colab.ipynb](notebooks/Sentiment_Distillation_Kaggle_Colab.ipynb)
to Kaggle (GPU + Internet enabled) or Colab (GPU runtime), then run cells in order.
By default, the notebook clones this private repository and uses its source and datasets.
In Kaggle, open **Add-ons → Secrets**, add `GITHUB_TOKEN` with a GitHub personal access token as its value,
and enable that secret for the notebook. A fine-grained token should select this repository with
**Contents: Read-only** permission. Set `PRIVATE_REPO` / `PRIVATE_BRANCH` in the configuration cell.
The Kaggle branch reads the token using:

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GITHUB_TOKEN")
```

Alternatively, set `GITHUB_AUTH='prompt'` to enter the token in a hidden input at runtime.
Colab supports its Secrets panel with Notebook access enabled, or the same hidden prompt.
The token is passed only through a temporary Git subprocess environment and an askpass helper;
it is not stored in the remote URL, source, credentials files or logs. Re-running the clone cell updates
a clean checkout using fast-forward only; local changes cause it to stop.

Set `USE_PRIVATE_REPO=False` to use bundled evaluation source without GitHub access; in that mode
benchmark data is downloaded from a fixed public upstream commit. The notebook does not implement fine-tuning.

Notebook defaults: Qwen 1.5B, SST2, first 32 test examples, batch size 1, four demonstrations,
2,048 input tokens. Set `MAX_SAMPLES=0` for a full dataset. This smoke configuration is not a full SentiBench score.

The notebook creates a separate venv that can access platform packages to reuse the existing CUDA-enabled
PyTorch. Project packages are installed into that venv, and all evaluators run in its subprocesses.
This is partial isolation, not a container. A pip dry-run blocks changes to PyTorch/NVIDIA/Triton packages.
Runtime checks validate the evaluation dependency closure, imports, tiny-model GPU inference/generation,
and optional 4-bit operations before the real evaluation starts. Logs and an environment report are exported.
The original `requirements.txt` is retained for reference; the notebook uses `requirements-notebook.txt`.

Evaluation saves atomic prediction checkpoints after each batch. Restore those files into `OUTPUT` to resume
in a new session with identical data, settings, source and package versions. Save Kaggle outputs or download
the results ZIP before the session ends; on Colab, use a mounted Drive output directory or download the ZIP.
GPU availability, quotas and session lifetimes remain controlled by the platform.

The patched scripts select FP16/BF16 automatically, accept Hugging Face IDs or local model directories,
honor input token limits, support optional NF4, and record experiment settings. They warn when demonstrations
are truncated and reject truncating away the test sentence. `parse.py` separates settings and reports missing
datasets as N/A rather than zero. Precision, truncation and changed demonstrations may affect benchmark scores.

The bundled configuration has **not been executed on Kaggle/Colab or dependency-tested locally**;
the user will run the included live checks on Kaggle. Static source/notebook checks are performed locally.
To rebuild the standalone notebook after changing the evaluator, run `python tools/build_notebook.py`.

Platform references: [Kaggle](https://www.kaggle.com/docs/notebooks),
[Colab resource limits](https://research.google.com/colaboratory/faq.html).

---

This repository contains the official datasets, and evaluation scripts for the paper:

**Targeted Distillation for Sentiment Analysis**
*Yice Zhang, Guangyu Xie, Jingjie Lin, Jianzhu Bao, Qianlong Wang, Xi Zeng, Ruifeng Xu (2025)*[[arXiv]](https://arxiv.org/abs/2503.03225)

Our work introduces a **targeted distillation framework** designed to transfer sentiment-related capabilities from advanced LLMs into compact language models. The project also provides **SentiBench**, a comprehensive benchmark for evaluating sentiment-analysis performance across diverse task types.

---

## 🚀 Abstract

This paper explores targeted distillation methods for sentiment analysis, aiming to build compact and practical models that preserve strong and generalizable sentiment analysis capabilities. To this end, we conceptually decouple the distillation target into knowledge and alignment and accordingly propose a two-stage distillation framework. Moreover, we introduce SentiBench, a comprehensive and systematic sentiment analysis benchmark that covers a diverse set of tasks across 12 datasets. We evaluate a wide range of models on this benchmark. Experimental results show that our approach substantially enhances the performance of compact models across diverse sentiment analysis tasks, and the resulting models demonstrate strong generalization to unseen tasks, showcasing robust competitiveness against existing small-scale models.


## 📊 SentiBench Benchmark

We introduce **SentiBench**, a sentiment analysis benchmark covering:

* **3 Task Categories**
* **12 Datasets**
* Including basic sentiment analysis, multifaceted sentiment analysis, and fine-grained sentiment analysis.

![alt text](./img/statistic.jpg)

---

## 📁 Project Structure

```
.
├── bash/                         # Evaluation scripts
│   ├── fsa.sh
│   ├── multifaced.sh
│   ├── multi_task_fsa.sh
│   └── multi_task_multifaced.sh
├── datasets/                     # SentiBench datasets
│   ├── absa/                     # Fine-grained sentiment analysis (FSA)
│   ├── multifaced/               # Multi-faceted sentiment analysis (MSA)
│   └── sc/                       # Basic sentiment analysis (BSA)
├── fsa.py                        # FSA evaluation entry
├── icl_multifaced.py             # MFSA evaluation entry
├── model_name.json               # Model version-to-path mapping
├── model_weights/                # Downloaded distilled models
│   ├── llama-3-1B-sentiment-distillation-v1
│   ├── llama-3-3B-sentiment-distillation-v1
│   └── Qwen2.5-1.5B-sentiment-distillation-v1
├── output/
│   └── result.txt                # Final results
├── parse.py                      # Result parser
├── parse_utils/                  # Parsing utilities for different tasks
│   ├── acsa.py
│   ├── asqp.py
│   ├── atsa.py
│   ├── coqe.py
│   ├── multifaced.py
│   └── ssa.py
├── utils.py
└── README.md
```

---


## **Training Corpus**

Our targeted distillation corpus: [https://huggingface.co/datasets/zhang-yice/sentiment-distillation-v1](https://huggingface.co/datasets/zhang-yice/sentiment-distillation-v1)

### **Released Distilled Models**

| Model                                  | Base Model   | HuggingFace                                                                                                                                          |
| -------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| llama-3-1B-sentiment-distillation-v1   | Llama-3-1B-Instruct   | [https://huggingface.co/zhang-yice/llama-3-1B-sentiment-distillation-v1](https://huggingface.co/zhang-yice/llama-3-1B-sentiment-distillation-v1)     |
| Qwen2.5-1.5B-sentiment-distillation-v1 | Qwen2.5-1.5B-Instruct | [https://huggingface.co/zhang-yice/Qwen2.5-1.5B-sentiment-distillation-v1](https://huggingface.co/zhang-yice/Qwen2.5-1.5B-sentiment-distillation-v1) |
| llama-3-3B-sentiment-distillation-v1   | Llama-3-3B-Instruct   | [https://huggingface.co/zhang-yice/llama-3-3B-sentiment-distillation-v1](https://huggingface.co/zhang-yice/llama-3-3B-sentiment-distillation-v1)     |

Download the corresponding model and place it into:

```
./model_weights/{model_name}
```

---

## 🧪 Evaluation

### 1️⃣ Download distilled models

Download the distilled model from huggingface and place each model under:

```
model_weights/
    ├── Qwen2.5-1.5B-sentiment-distillation-v1/
    ├── llama-3-1B-sentiment-distillation-v1/
    └── llama-3-3B-sentiment-distillation-v1/
```

Ensure `model_name.json` correctly maps the model version to its path.

---

### 2️⃣ Run Evaluation on SentiBench

#### **Qwen2.5-1.5B-sentiment-distillation-v1**

```bash
# Fine-grained Sentiment Analysis (FSA) 
bash/multi_task_fsa.sh -c 0 -b Qwen2.5-1.5B-sentiment-distillation-v1 -v Qwen2.5-1.5B-sentiment-distillation-v1 -z 16

# Multi-faceted Sentiment (MSA) + Basic Sentiment Analysis (BSA)
bash/multi_task_multifaced.sh -c 1 -b Qwen2.5-1.5B-sentiment-distillation-v1 -v Qwen2.5-1.5B-sentiment-distillation-v1
```

#### **llama-3-1B-sentiment-distillation-v1**

```bash
bash/multi_task_fsa.sh -c 0 -b llama-3-1B-sentiment-distillation-v1 -v llama-3-1B-sentiment-distillation-v1 -z 16
bash/multi_task_multifaced.sh -c 1 -b llama-3-1B-sentiment-distillation-v1 -v llama-3-1B-sentiment-distillation-v1
```

#### **llama-3-3B-sentiment-distillation-v1**

```bash
bash/multi_task_fsa.sh -c 0 -b llama-3-3B-sentiment-distillation-v1 -v llama-3-3B-sentiment-distillation-v1 -z 16
bash/multi_task_multifaced.sh -c 1 -b llama-3-3B-sentiment-distillation-v1 -v llama-3-3B-sentiment-distillation-v1
```

---

### 3️⃣ Parse Evaluation Results

```bash
python parse.py
```

Detailed results will appear in:

```
output/result.txt
```

---

## 📄 Citation

If you use our code, datasets, or models, please cite:

```bibtex
@inproceedings{zhang-etal-2025-targeted,
    title = "Targeted Distillation for Sentiment Analysis",
    author = "Zhang, Yice  and
      Xie, Guangyu  and
      Lin, Jingjie  and
      Bao, Jianzhu  and
      Wang, Qianlong  and
      Zeng, Xi  and
      Xu, Ruifeng",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1127/",
    doi = "10.18653/v1/2025.emnlp-main.1127",
    pages = "22169--22192",
    ISBN = "979-8-89176-332-6",
}
```
