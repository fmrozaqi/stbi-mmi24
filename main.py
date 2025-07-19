from fastapi import FastAPI
from search_embeddings import EmbeddingSearcher

app = FastAPI()
searcher = EmbeddingSearcher()

@app.get("/search/{query}")
def search(query: str):
    results = searcher.search_similar(query)
    return {"message": results}
