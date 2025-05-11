from functools import lru_cache
from typing import Dict, Any


class _InMemoryFeatureStore(dict):
    """TODO: NOT PROD READY"""


@lru_cache(maxsize=1)
def get_feature_store() -> _InMemoryFeatureStore:
    return _InMemoryFeatureStore()


def get_user_features(user_id: str) -> Dict[str, Any]:
    store = get_feature_store()
    return store.get(user_id, {})
