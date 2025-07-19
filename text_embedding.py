import json
from tqdm import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class TextEmbeddingIndexer:
    def __init__(self, es_host="http://localhost:9200", index_name="text_embeddings", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the TextEmbeddingIndexer.
        
        Args:
            es_host (str): Elasticsearch host URL
            index_name (str): Name of the Elasticsearch index
            model_name (str): Name of the sentence-transformer model to use
        """
        self.es_host = es_host
        self.index_name = index_name
        self.model = SentenceTransformer(model_name)
        self.es = Elasticsearch(es_host)
        
    def create_index(self):
        """Create the Elasticsearch index with mapping for text and embeddings."""
        if not self.es.indices.exists(index=self.index_name):
            # Define index mapping with text and dense_vector fields
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.model.get_sentence_embedding_dimension(),
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {"type": "object"}
                    }
                }
            }
            
            # Create the index
            self.es.indices.create(index=self.index_name, body=mapping)
            print(f"Created index '{self.index_name}'")
        else:
            print(f"Index '{self.index_name}' already exists")
    
    def generate_embedding(self, text):
        """
        Generate embedding for a text.
        
        Args:
            text (str): Input text
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        return self.model.encode(text)
    
    def index_document(self, text, doc_id=None, metadata=None):
        """
        Index a single document with its embedding.
        
        Args:
            text (str): Text to embed and index
            doc_id (str, optional): Document ID
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Document ID
        """
        embedding = self.generate_embedding(text)
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        # Prepare document
        doc = {
            "text": text,
            "embedding": embedding_list,
            "metadata": metadata or {}
        }
        
        # Index the document
        response = self.es.index(
            index=self.index_name,
            id=doc_id,
            document=doc
        )
        
        return response["_id"]
    
    def bulk_index(self, texts, ids=None, metadata_list=None):
        """
        Index multiple documents in bulk.
        
        Args:
            texts (list): List of texts to embed and index
            ids (list, optional): List of document IDs
            metadata_list (list, optional): List of metadata dictionaries
            
        Returns:
            list: List of document IDs
        """
        if ids is None:
            ids = [None] * len(texts)
        
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        doc_ids = []
        
        # Process in batches to avoid memory issues with large datasets
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Indexing batches"):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Generate embeddings for the batch
            embeddings = self.model.encode(batch_texts)
            
            # Prepare bulk indexing operations
            operations = []
            for j, (text, embedding, doc_id, metadata) in enumerate(zip(batch_texts, embeddings, batch_ids, batch_metadata)):
                # Add index action
                action = {"index": {"_index": self.index_name}}
                if doc_id:
                    action["index"]["_id"] = doc_id
                
                operations.append(action)
                
                # Add document source
                doc = {
                    "text": text,
                    "embedding": embedding.tolist(),
                    "metadata": metadata
                }
                operations.append(doc)
            
            # Perform bulk indexing
            resp = self.es.bulk(operations=operations)
            
            # Collect document IDs
            for item in resp["items"]:
                doc_ids.append(item["index"]["_id"])
        
        return doc_ids
    
    def search_similar(self, query_text, top_k=5):
        """
        Search for documents similar to the query text.
        
        Args:
            query_text (str): Query text
            top_k (int): Number of results to return
            
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
        
        # Execute the search
        response = self.es.search(
            index=self.index_name,
            body=search_query
        )
        
        # Extract results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "text": hit["_source"]["text"],
                "metadata": hit["_source"]["metadata"]
            })
        
        return results


# Example usage
if __name__ == "__main__":
    # Sample texts
    sample_texts = [
        "Elasticsearch is a distributed, RESTful search and analytics engine.",
        "Python is a programming language that lets you work quickly and integrate systems effectively.",
        "Natural Language Processing (NLP) is a field of AI that gives machines the ability to read and understand human language.",
        "Vector embeddings are numerical representations of words or documents in a continuous vector space.",
        "Docker is a platform for developing, shipping, and running applications in containers."
    ]
    
    # Initialize the indexer
    indexer = TextEmbeddingIndexer()
    
    # Create the index
    indexer.create_index()
    
    # Index the sample texts
    print("Indexing sample texts...")
    doc_ids = indexer.bulk_index(
        texts=sample_texts,
        metadata_list=[{"source": "example", "category": f"category_{i}"} for i in range(len(sample_texts))]
    )
    
    print(f"Indexed {len(doc_ids)} documents")
    
    # Search for similar documents
    query = "AI and machine learning technologies"
    print(f"\nSearching for documents similar to: '{query}'")
    
    results = indexer.search_similar(query)
    
    print("\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}")
        print(f"   Metadata: {json.dumps(result['metadata'])}")
        print()
