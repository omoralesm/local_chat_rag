from langchain.vectorstores.chroma import Chroma
from get_embedding_function import LlamaCppEmbeddingFunction

CHROMA_PATH = "D:/LLM/Chroma"
MODEL_EMBEDDING_PATH = "D:/LLM/Models/all-MiniLM-L6-v2.F16.gguf"
PROMPT_TEMPLATE = """
User wants to know [{question}] Use the extracted passages from our knowledge base 
(below) to form the best answer you can. Show the final answer clearly.
[{context}]
"""

def query_rag(search_query: str):
    # Prepare the DB.
    embedding_function = LlamaCppEmbeddingFunction(model_path=MODEL_EMBEDDING_PATH)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    query_vector = db.similarity_search_with_score(search_query, k=4)
    
    return PROMPT_TEMPLATE.format(
            context = "\n\n---\n\n".join([doc.page_content for doc, _score in query_vector]),
            question = search_query      
        )
    