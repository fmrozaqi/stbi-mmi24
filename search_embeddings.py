import json
import argparse
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class EmbeddingSearcher:
    def __init__(self, es_host="http://localhost:9200", index_name="text_embeddings", model_name="all-MiniLM-L6-v2"):
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
    
    def search_similar(self, query_text, top_k=5, min_score=0.0, filter_query=None):
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
    
    def search_by_metadata(self, metadata_filter, top_k=10):
        """
        Search for documents by metadata.
        
        Args:
            metadata_filter (dict): Filter criteria for metadata fields
            top_k (int): Number of results to return
            
        Returns:
            list: List of matching documents
        """
        # Build the query
        query = {
            "query": {
                "bool": {
                    "must": []
                }
            },
            "size": top_k
        }
        
        # Add metadata filters
        for key, value in metadata_filter.items():
            query["query"]["bool"]["must"].append({
                "match": {
                    f"metadata.{key}": value
                }
            })
        
        # Execute the search
        response = self.es.search(
            index=self.index_name,
            body=query
        )
        
        # Extract results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "text": hit["_source"]["text"],
                "metadata": hit["_source"].get("metadata", {})
            })
        
        return results
    
    def hybrid_search(self, query_text, top_k=5, text_boost=0.5, vector_boost=0.5):
        """
        Perform hybrid search combining text and vector similarity.
        
        Args:
            query_text (str): Query text
            top_k (int): Number of results to return
            text_boost (float): Weight for text search component
            vector_boost (float): Weight for vector search component
            
        Returns:
            list: List of matching documents
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query_text)
        
        # Build the query
        query = {
            "query": {
                "script_score": {
                    "query": {
                        "match": {
                            "text": {
                                "query": query_text,
                                "boost": text_boost
                            }
                        }
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, 'embedding') * {vector_boost}",
                        "params": {
                            "query_vector": query_embedding.tolist()
                        }
                    }
                }
            },
            "size": top_k
        }
        
        # Execute the search
        response = self.es.search(
            index=self.index_name,
            body=query
        )
        
        # Extract results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "text": hit["_source"]["text"],
                "metadata": hit["_source"].get("metadata", {})
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Search for similar documents in Elasticsearch')
    parser.add_argument('--query', type=str, required=True, help='Query text to search for')
    parser.add_argument('--host', type=str, default='http://localhost:9200', help='Elasticsearch host URL')
    parser.add_argument('--index', type=str, default='text_embeddings', help='Index name')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Sentence transformer model name')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum similarity score (0-1)')
    parser.add_argument('--category', type=str, help='Filter by category metadata')
    parser.add_argument('--source', type=str, help='Filter by source metadata')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid search (text + vector)')
    
    args = parser.parse_args()
    
    # Initialize the searcher
    searcher = EmbeddingSearcher(
        es_host=args.host,
        index_name=args.index,
        model_name=args.model
    )
    
    # Prepare metadata filter if provided
    metadata_filter = {}
    if args.category:
        metadata_filter['category'] = args.category
    if args.source:
        metadata_filter['source'] = args.source
    
    # Perform search
    if args.hybrid:
        print(f"Performing hybrid search for: '{args.query}'")
        results = searcher.hybrid_search(args.query, top_k=args.top_k)
    elif metadata_filter:
        print(f"Searching by metadata: {metadata_filter}")
        results = searcher.search_by_metadata(metadata_filter, top_k=args.top_k)
    else:
        print(f"Searching for documents similar to: '{args.query}'")
        results = searcher.search_similar(
            args.query, 
            top_k=args.top_k, 
            min_score=args.min_score
        )
    
    # Display results
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}")
        print(f"   Metadata: {json.dumps(result['metadata'])}")
        print()


if __name__ == "__main__":
    main()
