from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def encode_query(self, text: str):
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]