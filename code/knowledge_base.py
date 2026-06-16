import os
from typing import Any, Dict, List

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


class MyEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding adapter backed by SentenceTransformer."""

    def __init__(self, model):
        self.embedding_model = SentenceTransformer(model, trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:
        if hasattr(self.embedding_model, "encode_document"):
            return self.embedding_model.encode_document(input)
        return self.embedding_model.encode(input)


class KnowledgeBaseChroma:
    def __init__(self, policy_model: str = "qwen-32b", collection_name: str = "knowledge_base"):
        current_dir = os.path.dirname(__file__)
        kb_root = os.getenv("UKA_KB_ROOT", os.path.join(current_dir, "kb"))
        embedding_model = os.getenv(
            "UKA_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.client = chromadb.PersistentClient(path=os.path.join(kb_root, policy_model))
        self.embedding_function = MyEmbeddingFunction(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_known = self.client.get_or_create_collection(
            name="new_knowledges",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, texts: list, metadatas: list = None, ids: list = None):
        """Add documents and mirrored knowledge snippets to Chroma."""
        doc_count = self.collection.count()
        if ids is None:
            ids = [str(doc_count + i) for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        self.collection_known.add(
            ids=ids,
            documents=[item.get("source", "") for item in metadatas],
            metadatas=[{"source": "no"} for _ in texts],
        )

    def retrieve_topk(
        self,
        query: str,
        K: int = 5,
        metadata_filter: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        res = self.collection.query(
            query_texts=[query],
            n_results=K,
            where=metadata_filter,
        )
        return [
            {"text": doc, "score": float(dist), "metadata": meta}
            for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
        ]

    def retrieve_topk_known(
        self,
        query: str,
        K: int = 3,
        metadata_filter: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        res = self.collection_known.query(
            query_texts=[query],
            n_results=K,
            where=metadata_filter,
        )
        return [
            {"text": doc, "score": float(dist), "metadata": meta}
            for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
        ]
