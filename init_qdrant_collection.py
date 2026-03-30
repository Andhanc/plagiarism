"""
Создаёт коллекцию Qdrant для векторного плагиата (если ещё нет).
Те же параметры, что в showcase.py: cosine, размер 312 (rubert-tiny2), INT8 quantization.

Использование:
  export QDRANT_URL=http://localhost:6333
  export QDRANT_COLLECTION=university_docs
  python init_qdrant_collection.py
"""
from __future__ import annotations

import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

VECTOR_SIZE = 312  # cointegrated/rubert-tiny2


def main():
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    name = os.environ.get("QDRANT_COLLECTION", "university_docs")
    client = QdrantClient(url=url, prefer_grpc=False, timeout=30.0)
    if client.collection_exists(name):
        print(f"Collection '{name}' already exists")
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
        ),
    )
    print(f"Created collection '{name}' (dim={VECTOR_SIZE}, cosine, INT8)")


if __name__ == "__main__":
    main()
