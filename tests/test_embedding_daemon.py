from services.embedding_daemon import _resolve_embedding_count


def test_resolve_embedding_count_prefers_total_embeddings():
    stats = {
        "total_embeddings": 12,
        "embeddings_count": 5,
        "embeddings": 3,
    }
    assert _resolve_embedding_count(stats) == 12


def test_resolve_embedding_count_handles_alternate_keys():
    assert _resolve_embedding_count({"embeddings_count": 7}) == 7
    assert _resolve_embedding_count({"embeddings": 9}) == 9
    assert _resolve_embedding_count({"embeddings": "11"}) == 11
    assert _resolve_embedding_count({"embeddings": None}) == 0
    assert _resolve_embedding_count({}) == 0
