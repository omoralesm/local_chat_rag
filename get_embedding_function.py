from llama_cpp import Llama
import chromadb.utils.embedding_functions as embedding_functions

class LlamaCppEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_path: str, embedding: bool = True):
        self.model = Llama(model_path=model_path, embedding=embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            # The create_embedding method returns a dictionary, extract the embedding vector
            result = self.model.create_embedding(text)
            embeddings.append(result['data'][0]['embedding'])
        return embeddings
    
    def embed_query(self, query: str) -> list[float]:
        result = self.model.create_embedding(query)
        return result['data'][0]['embedding']
        