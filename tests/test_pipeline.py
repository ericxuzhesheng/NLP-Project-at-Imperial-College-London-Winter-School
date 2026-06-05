from biomed_repr.representations import CoOccurrenceEmbeddings
from biomed_repr.tokenization import regex_tokenize, tokenize_lines


def test_regex_tokenizer_keeps_biomedical_terms():
    tokens = regex_tokenize("SARS-CoV-2 binds ACE2 receptors in COVID-19.")
    assert "sars-cov-2" in tokens
    assert "ace2" in tokens
    assert "covid-19" in tokens


def test_cooccurrence_embeddings_return_similar_words():
    sentences = tokenize_lines(
        [
            "coronavirus infection causes fever and cough",
            "influenza infection causes fever and fatigue",
            "vaccine therapy prevents coronavirus infection",
            "antiviral therapy treats influenza infection",
        ]
    )
    model = CoOccurrenceEmbeddings(vector_size=4, window_size=2, min_count=1).fit(sentences)
    neighbors = dict(model.most_similar("coronavirus", topn=5))
    assert "infection" in neighbors or "vaccine" in neighbors
