from biomed_repr.representations import CoOccurrenceEmbeddings
from biomed_repr.tokenization import regex_tokenize, tokenize_lines
from biomed_repr.bpe import SimpleBPE
from biomed_repr.ngram import NGramLanguageModel
from biomed_repr.word2vec import SkipGramNegativeSampling


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


def test_simple_bpe_learns_subword_merges():
    tokenizer = SimpleBPE.train(
        [
            "coronavirus coronavirus vaccine",
            "coronavirus infection vaccine",
        ],
        num_merges=10,
        min_frequency=2,
    )
    pieces = tokenizer.encode("coronavirus vaccine")
    assert pieces
    assert all(piece != "</w>" for piece in pieces)


def test_ngram_predicts_next_word():
    sentences = tokenize_lines(
        [
            "covid patients develop fever",
            "covid patients develop cough",
            "influenza patients develop fever",
        ]
    )
    model = NGramLanguageModel(n=3, min_count=1, vector_size=4).fit(sentences)
    predictions = dict(model.predict_next(["covid", "patients"], topn=5))
    assert "develop" in predictions


def test_sgns_smoke_training():
    sentences = tokenize_lines(
        [
            "coronavirus infection fever",
            "coronavirus vaccine therapy",
            "influenza infection fever",
            "influenza antiviral therapy",
        ]
    )
    model = SkipGramNegativeSampling(
        vector_size=4,
        window_size=1,
        negative_samples=2,
        min_count=1,
        epochs=1,
        seed=7,
    ).fit(sentences)
    assert model.most_similar("coronavirus", topn=2)
