import json
import argparse
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class EmbeddingSearcher:
    def __init__(self, es_host="http://localhost:9200", index_name="semantic_scholar", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingSearcher.
        
        Args:
            es_host (str): Elasticsearch host URL
            index_name (str): Name of the Elasticsearch index to search
            model_name (str): Name of the sentence-transformer model to use
        """
        self.es_host = es_host
        self.index_name = index_name
        self.model = SentenceTransformer(model_name)
        self.es = Elasticsearch(es_host)
        
        # Check if index exists
        if not self.es.indices.exists(index=self.index_name):
            raise ValueError(f"Index '{self.index_name}' does not exist. Please create it first.")
    
    def generate_embedding(self, text):
        """
        Generate embedding for a text.
        
        Args:
            text (str): Input text
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        return self.model.encode(text)
    
    def search_similar(self, query_text, top_k=20, min_score=0.5, filter_query=None):
        """
        Search for documents similar to the query text.
        
        Args:
            query_text (str): Query text
            top_k (int): Number of results to return
            min_score (float): Minimum similarity score (0-1)
            filter_query (dict, optional): Elasticsearch filter query to apply
            
        Returns:
            list: List of similar documents with scores
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query_text)
        
        # Prepare the search query
        search_query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": top_k,
                "num_candidates": top_k * 2
            },
            "_source": ["text", "metadata"]
        }
        
        # Add filter if provided
        if filter_query:
            search_query["post_filter"] = filter_query
        
        # Execute the search
        response = self.es.search(
            index=self.index_name,
            body=search_query
        )
        
        # Extract results
        results = []
        for hit in response["hits"]["hits"]:
            # Skip results below min_score
            if hit["_score"] < min_score:
                continue
                
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "text": hit["_source"]["text"],
                "metadata": hit["_source"].get("metadata", {})
            })
        
        return results
