# 生物医学领域词表示 / Word Representation in the Biomedical Domain

## 项目概览

本项目来自 Imperial College London Data Science and AI School 的 NLP 课程项目，目标是在 CORD-19 等生物医学文本上构建可解释、可复现的词表示流水线。原始实验记录保留在 `word_representations_biomedical.ipynb`；当前仓库已重构为模块化 Python 代码，便于复用、测试和扩展。

核心能力包括：

- 从 CORD-19 `metadata.csv`、解析后的 JSON 或普通文本中抽取语料。
- 使用轻量正则分词作为默认方案，并保留 NLTK / BERT tokenizer 的可选入口。
- 基于共现矩阵和 SPPMI/SVD 训练无需外部模型下载的词向量。
- 输出语义相似词，并可选生成 t-SNE 可视化。
- 保留课程报告、讲义和已有 tokenized 数据作为项目证据与复现实验素材。

## 仓库结构

```text
.
├── src/biomed_repr/              # 重构后的可复用源码包
│   ├── corpus.py                 # CORD-19/CSV/纯文本语料读取
│   ├── tokenization.py           # 正则、NLTK、BERT 分词入口
│   ├── representations.py        # 共现词向量模型
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
2. **分词**：默认正则分词保留 `sars-cov-2`、`covid-19` 等生物医学连字符词；NLTK 和 BERT 分词作为可选增强。
3. **词表示**：`CoOccurrenceEmbeddings` 使用窗口共现统计、shifted positive PMI 和 SVD 生成固定维度向量，适合课程项目和本地快速复现。
4. **探索分析**：支持查询目标词相似词、导出 CSV，并在安装可视化依赖后生成 t-SNE 图。

## 测试

```powershell
$env:PYTHONPATH="src"
python -m pytest
```

测试覆盖了生物医学术语分词和共现词向量的基本相似词查询。

## 致谢

本项目基于 Imperial College London Data Science and AI School NLP 课程要求完成。数据来源参考 [CORD-19](https://www.semanticscholar.org/cord19)。

## 许可证

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

---

## Overview

This project was developed for the NLP component of the Imperial College London Data Science and AI School programme. It builds an interpretable and reproducible word-representation pipeline for biomedical corpora such as CORD-19. The original exploratory notebook is preserved in `word_representations_biomedical.ipynb`; the main implementation has been refactored into a modular Python package.

Key features:

- Extract corpora from CORD-19 `metadata.csv`, parsed JSON files, or plain text.
- Use a lightweight regex tokenizer by default, with optional NLTK and BERT tokenizer backends.
- Train local word vectors with a co-occurrence matrix, shifted positive PMI, and SVD.
- Export nearest-neighbor terms and optionally create t-SNE visualizations.
- Keep course reports, lecture notes, and generated tokenized files as reproducibility artifacts.

## Repository Layout

```text
.
├── src/biomed_repr/              # Refactored reusable source package
│   ├── corpus.py                 # CORD-19/CSV/plain-text corpus readers
│   ├── tokenization.py           # Regex, NLTK, and BERT tokenizers
│   ├── representations.py        # Co-occurrence embedding model
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
2. **Tokenization**: the default regex tokenizer preserves biomedical hyphenated terms such as `sars-cov-2` and `covid-19`; NLTK and BERT tokenizers are available as optional alternatives.
3. **Word representations**: `CoOccurrenceEmbeddings` creates fixed-size vectors from windowed co-occurrence statistics, shifted positive PMI, and SVD.
4. **Exploration**: utilities support nearest-neighbor lookup, CSV export, and optional t-SNE plots.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m pytest
```

The tests cover biomedical tokenization and a smoke test for co-occurrence similarity search.

## Acknowledgements

Developed as part of the Natural Language Processing course at Imperial College London Data Science and AI School. Dataset reference: [CORD-19](https://www.semanticscholar.org/cord19).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
