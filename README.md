# Word Representation in the Biomedical Domain

## Overview

This project demonstrates how **Natural Language Processing (NLP)** techniques can be applied to large-scale **biomedical text** to learn and explore **word representations**.  
Using the [CORD-19](https://www.semanticscholar.org/cord19) dataset (500K+ scholarly articles), we build domain-specific embeddings to capture semantic relationships between biomedical terms and visualize them for biomedical knowledge discovery.

---

## Methodology

### 1. Dataset Processing

- **Source:** [CORD-19](https://www.semanticscholar.org/cord19) containing over 500K scholarly articles, including 200K+ with full text.
- **Preprocessing:** Extracted and cleaned biomedical research content from JSON and XML files.

### 2. Tokenization

Applied multiple tokenization strategies to prepare biomedical text for modeling:

- **Regex-based split:** Used Python `split()` with regular expressions for basic token segmentation.
- **NLTK Tokenizer:** Applied NLTK's word tokenization to handle punctuation and basic linguistic rules.
- **Byte-Pair Encoding (BPE):** Implemented subword segmentation to handle rare biomedical terms.
- **Custom BPE:** Built a domain-specific BPE vocabulary to better represent biomedical entities.

### 3. Word Representation Modeling

Trained embeddings using multiple methods:

- **N-gram Language Modeling:** Captures local context by predicting next words based on previous n words.
- **Skip-gram with Negative Sampling:** Learns word vectors by predicting surrounding words for each target word while reducing noise.
- **Contextualized Word Representation (MLM):** Uses Masked Language Modeling to generate embeddings that depend on sentence context (e.g., BERT-style models).

### 4. Visualization & Analysis

Explored embeddings using:

- **t-SNE Dimensionality Reduction:** Projects high-dimensional vectors into 2D for visualization.
- **Biomedical Entity Clustering:** Groups semantically related biomedical terms (e.g., diseases, treatments, proteins).
- **Co-occurrence Analysis:** Detects term relationships based on frequency of appearing together in context.
- **Semantic Similarity Measurement:** Computes cosine similarity to identify related biomedical concepts.

---

## Key Achievements

- Built **customized tokenization and embedding models** for biomedical text.
- Generated **domain-specific word vectors** capturing semantic relationships.
- Created **visual analytics tools** for biomedical literature exploration.
- Demonstrated applications in **entity clustering, similarity search, and knowledge mining**.

---

## Acknowledgements

Developed as part of the **Natural Language Processing** course at **Imperial College London Data Science and AI School**.  
Dataset: [CORD-19](https://www.semanticscholar.org/cord19)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
