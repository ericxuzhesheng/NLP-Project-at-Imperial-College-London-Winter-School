# 生物医学领域词表示 / Word Representation in the Biomedical Domain

## 项目概览

本项目来自 Imperial College London Data Science and AI School 的 NLP 课程项目，目标是在 CORD-19 等生物医学文本上构建可解释、可复现的词表示流水线。原始实验记录保留在 `word_representations_biomedical.ipynb`；当前仓库已重构为模块化 Python 代码，便于复用、测试和扩展。

核心能力包括：

- 从 CORD-19 `metadata.csv`、解析后的 JSON 或普通文本中抽取语料。
- 支持正则、NLTK、BERT tokenizer、自定义 SimpleBPE 和 Hugging Face BPE。
- 支持共现 SPPMI/SVD、N-gram、手写 Skip-gram Negative Sampling、Gensim Word2Vec、BERT MLM + LoRA。
- 输出语义相似词、N-gram next-word prediction、CSV/XLSX 结果，并可选生成普通或生物医学实体分组 t-SNE 可视化。
- 保留课程报告、讲义和已有 tokenized 数据作为项目证据与复现实验素材。

## 仓库结构

```text
.
├── src/biomed_repr/              # 重构后的可复用源码包
│   ├── corpus.py                 # CORD-19/CSV/纯文本语料读取
│   ├── tokenization.py           # 正则、NLTK、BERT 分词入口
│   ├── bpe.py                    # SimpleBPE 与 Hugging Face BPE
│   ├── ngram.py                  # N-gram 语言模型
│   ├── representations.py        # 共现词向量模型
│   ├── word2vec.py               # 手写 SGNS 与 Gensim Word2Vec
│   ├── mlm.py                    # BERT MLM + LoRA 微调入口
│   ├── explore.py                # 相似词与 t-SNE 探索工具
│   └── cli.py                    # 命令行入口
├── tests/                        # 轻量单元测试
├── data/                         # 已生成的分词结果和图像
├── word_representations_biomedical.ipynb
├── nlp report.pdf
└── README.md
```

## 快速开始

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m biomed_repr.cli train --input data/2_nlkt_tokenized.txt --output outputs/cooccurrence.pkl --limit 200 --max-vocab 500 --vector-size 50 --min-count 1
python -m biomed_repr.cli similar --model outputs/cooccurrence.pkl --word coronavirus --output outputs/coronavirus_similar.csv
```

## 主要功能命令

```powershell
# 自定义 BPE：纯 Python 实现
python -m biomed_repr.cli bpe-train --backend simple --input data/2_nlkt_tokenized.txt --output outputs/simple_bpe.json --limit 200 --num-merges 100
python -m biomed_repr.cli bpe-tokenize --backend simple --model outputs/simple_bpe.json --input data/2_nlkt_tokenized.txt --output outputs/simple_bpe_tokenized.txt --limit 20

# Hugging Face BPE：需要 tokenizers
python -m biomed_repr.cli bpe-train --backend hf --input outputs/corpus.txt --output outputs/hf_bpe.json --vocab-size 30000

# N-gram 语言模型和 next-word prediction
python -m biomed_repr.cli train --model-type ngram --input data/2_nlkt_tokenized.txt --output outputs/ngram.pkl --limit 200 --min-count 1
python -m biomed_repr.cli predict-next --model outputs/ngram.pkl --context "covid patients"

# 手写 Skip-gram Negative Sampling
python -m biomed_repr.cli train --model-type sgns --input data/2_nlkt_tokenized.txt --output outputs/sgns.pkl --limit 200 --max-vocab 500 --min-count 1 --epochs 1

# Gensim Word2Vec：需要 gensim
python -m biomed_repr.cli train-word2vec --input data/2_nlkt_tokenized.txt --output outputs/word2vec.model --limit 20000

# BERT MLM + LoRA：需要 transformers、datasets、peft，并会下载/读取预训练模型
python -m biomed_repr.cli train-mlm-lora --input outputs/corpus.txt --output-dir outputs/mlm_lora --max-lines 1000 --epochs 1

# 普通 t-SNE 与生物医学实体彩色 t-SNE：需要 scikit-learn、matplotlib
python -m biomed_repr.cli plot --model outputs/cooccurrence.pkl --output outputs/tsne.png --max-words 300
python -m biomed_repr.cli plot --model outputs/cooccurrence.pkl --output outputs/entity_tsne.png --max-words 300 --entities

# 相似词导出 CSV 或 XLSX；XLSX 需要 openpyxl
python -m biomed_repr.cli similar --model outputs/cooccurrence.pkl --word coronavirus --output outputs/similar_words.xlsx
python -m biomed_repr.cli entity-inventory --model outputs/cooccurrence.pkl --output outputs/entities.csv
```

如果需要从原始 CORD-19 文件重新构建语料：

```powershell
$env:PYTHONPATH="src"
python -m biomed_repr.cli prepare-corpus --source-type metadata --input path/to/metadata.csv --output outputs/corpus.txt --limit 10000
python -m biomed_repr.cli tokenize --input outputs/corpus.txt --output outputs/tokenized.txt --tokenizer regex
python -m biomed_repr.cli train --input outputs/tokenized.txt --output outputs/cooccurrence.pkl
```

可选 t-SNE 可视化需要安装 `scikit-learn` 和 `matplotlib`：

```powershell
python -m biomed_repr.cli plot --model outputs/cooccurrence.pkl --output outputs/tsne.png --max-words 300
```

## 方法说明

1. **语料处理**：`corpus.py` 支持读取 CORD-19 metadata、解析后的 JSON 文件夹，以及一行一篇文档的纯文本语料。
2. **分词**：默认正则分词保留 `sars-cov-2`、`covid-19`、`ace2` 等生物医学术语；NLTK、BERT、BPE 作为可选增强。
3. **词表示**：覆盖课程 notebook 的主要建模路线，包括 N-gram、Skip-gram Negative Sampling、Word2Vec、共现向量和 MLM-LoRA。
4. **探索分析**：支持相似词、共现近邻、实体词表盘点、CSV/XLSX 导出，以及普通/实体分组 t-SNE。

## 测试

```powershell
$env:PYTHONPATH="src"
python -m pytest
```

测试覆盖了生物医学术语分词、共现词向量、SimpleBPE、N-gram 和手写 SGNS 的轻量 smoke path。

## 致谢

本项目基于 Imperial College London Data Science and AI School NLP 课程要求完成。数据来源参考 [CORD-19](https://www.semanticscholar.org/cord19)。

## 许可证

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

---

## Overview

This project was developed for the NLP component of the Imperial College London Data Science and AI School programme. It builds an interpretable and reproducible word-representation pipeline for biomedical corpora such as CORD-19. The original exploratory notebook is preserved in `word_representations_biomedical.ipynb`; the main implementation has been refactored into a modular Python package.

Key features:

- Extract corpora from CORD-19 `metadata.csv`, parsed JSON files, or plain text.
- Support regex, NLTK, BERT tokenizer, custom SimpleBPE, and Hugging Face BPE.
- Train co-occurrence SPPMI/SVD, N-gram, from-scratch Skip-gram Negative Sampling, Gensim Word2Vec, and BERT MLM + LoRA models.
- Export nearest-neighbor terms, N-gram next-word predictions, CSV/XLSX outputs, and regular or biomedical entity t-SNE visualizations.
- Keep course reports, lecture notes, and generated tokenized files as reproducibility artifacts.

## Repository Layout

```text
.
├── src/biomed_repr/              # Refactored reusable source package
│   ├── corpus.py                 # CORD-19/CSV/plain-text corpus readers
│   ├── tokenization.py           # Regex, NLTK, and BERT tokenizers
│   ├── bpe.py                    # SimpleBPE and Hugging Face BPE
│   ├── ngram.py                  # N-gram language model
│   ├── representations.py        # Co-occurrence embedding model
│   ├── word2vec.py               # From-scratch SGNS and Gensim Word2Vec
│   ├── mlm.py                    # BERT MLM + LoRA fine-tuning entry
│   ├── explore.py                # Similarity and t-SNE utilities
│   └── cli.py                    # Command line interface
├── tests/                        # Lightweight unit tests
├── data/                         # Existing tokenized outputs and figures
├── word_representations_biomedical.ipynb
├── nlp report.pdf
└── README.md
```

## Quick Start

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m biomed_repr.cli train --input data/2_nlkt_tokenized.txt --output outputs/cooccurrence.pkl --limit 200 --max-vocab 500 --vector-size 50 --min-count 1
python -m biomed_repr.cli similar --model outputs/cooccurrence.pkl --word coronavirus --output outputs/coronavirus_similar.csv
```

## Main Commands

```powershell
# Custom BPE: pure Python implementation
python -m biomed_repr.cli bpe-train --backend simple --input data/2_nlkt_tokenized.txt --output outputs/simple_bpe.json --limit 200 --num-merges 100
python -m biomed_repr.cli bpe-tokenize --backend simple --model outputs/simple_bpe.json --input data/2_nlkt_tokenized.txt --output outputs/simple_bpe_tokenized.txt --limit 20

# Hugging Face BPE: requires tokenizers
python -m biomed_repr.cli bpe-train --backend hf --input outputs/corpus.txt --output outputs/hf_bpe.json --vocab-size 30000

# N-gram language model and next-word prediction
python -m biomed_repr.cli train --model-type ngram --input data/2_nlkt_tokenized.txt --output outputs/ngram.pkl --limit 200 --min-count 1
python -m biomed_repr.cli predict-next --model outputs/ngram.pkl --context "covid patients"

# From-scratch Skip-gram Negative Sampling
python -m biomed_repr.cli train --model-type sgns --input data/2_nlkt_tokenized.txt --output outputs/sgns.pkl --limit 200 --max-vocab 500 --min-count 1 --epochs 1

# Gensim Word2Vec: requires gensim
python -m biomed_repr.cli train-word2vec --input data/2_nlkt_tokenized.txt --output outputs/word2vec.model --limit 20000

# BERT MLM + LoRA: requires transformers, datasets, and peft; it downloads or reads a pretrained model
python -m biomed_repr.cli train-mlm-lora --input outputs/corpus.txt --output-dir outputs/mlm_lora --max-lines 1000 --epochs 1

# Regular t-SNE and biomedical entity t-SNE: require scikit-learn and matplotlib
python -m biomed_repr.cli plot --model outputs/cooccurrence.pkl --output outputs/tsne.png --max-words 300
python -m biomed_repr.cli plot --model outputs/cooccurrence.pkl --output outputs/entity_tsne.png --max-words 300 --entities

# Similarity export to CSV or XLSX; XLSX requires openpyxl
python -m biomed_repr.cli similar --model outputs/cooccurrence.pkl --word coronavirus --output outputs/similar_words.xlsx
python -m biomed_repr.cli entity-inventory --model outputs/cooccurrence.pkl --output outputs/entities.csv
```

To rebuild the corpus from raw CORD-19 files:

```powershell
$env:PYTHONPATH="src"
python -m biomed_repr.cli prepare-corpus --source-type metadata --input path/to/metadata.csv --output outputs/corpus.txt --limit 10000
python -m biomed_repr.cli tokenize --input outputs/corpus.txt --output outputs/tokenized.txt --tokenizer regex
python -m biomed_repr.cli train --input outputs/tokenized.txt --output outputs/cooccurrence.pkl
```

Optional t-SNE visualization requires `scikit-learn` and `matplotlib`:

```powershell
python -m biomed_repr.cli plot --model outputs/cooccurrence.pkl --output outputs/tsne.png --max-words 300
```

## Methodology

1. **Corpus processing**: `corpus.py` reads CORD-19 metadata, parsed JSON directories, or plain text corpora.
2. **Tokenization**: the default regex tokenizer preserves biomedical terms such as `sars-cov-2`, `covid-19`, and `ace2`; NLTK, BERT, and BPE are available as optional alternatives.
3. **Word representations**: the implementation covers the notebook's main modeling routes: N-gram, Skip-gram Negative Sampling, Word2Vec, co-occurrence vectors, and MLM-LoRA.
4. **Exploration**: utilities support nearest-neighbor lookup, co-occurrence-style neighbors, biomedical entity inventory, CSV/XLSX export, and regular or entity-colored t-SNE plots.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m pytest
```

The tests cover biomedical tokenization plus smoke paths for co-occurrence vectors, SimpleBPE, N-gram, and from-scratch SGNS.

## Acknowledgements

Developed as part of the Natural Language Processing course at Imperial College London Data Science and AI School. Dataset reference: [CORD-19](https://www.semanticscholar.org/cord19).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
