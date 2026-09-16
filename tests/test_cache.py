from pydantic import BaseModel
from book2video.cache import StageCache


class X(BaseModel):
    value: int


def test_cache_roundtrip(tmp_path):
    cache = StageCache(tmp_path)
    key = cache.key("x", {"a": 1})
    cache.put("x", key, X(value=3))
    out = cache.get("x", key, X)
    assert out is not None
    assert out.value == 3
