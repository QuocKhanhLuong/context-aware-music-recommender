from src.features.pos_hmm_viterbi import TOY_TAGGED_CORPUS, train_hmm_pos_tagger, viterbi_decode, pos_tag_lyrics, extract_pos_features


def test_viterbi_returns_same_length_as_tokens():
    transition, emission, priors = train_hmm_pos_tagger(TOY_TAGGED_CORPUS)
    tokens = ["i", "dance", "slowly"]
    tags = viterbi_decode(tokens, transition, emission, priors)
    assert len(tags) == len(tokens)


def test_pos_features_have_expected_keys():
    tags = pos_tag_lyrics("i love calm music")
    features = extract_pos_features(tags)
    assert {"noun_ratio", "verb_ratio", "adjective_ratio", "adverb_ratio", "pronoun_ratio", "unique_pos_count"}.issubset(features)
