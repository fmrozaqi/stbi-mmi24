import json
from tqdm import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class TextEmbeddingIndexer:
    def __init__(self, es_host="http://localhost:9200", index_name="semantic_scholar", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the TextEmbeddingIndexer.
        
        Args:
            es_host (str): Elasticsearch host URL
            index_name (str): Name of the Elasticsearch index
            model_name (str): Name of the sentence-transformer model to use
        """
        self.es_host = es_host
        self.index_name = index_name
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
            return False
        else:
            print(f"Index '{self.index_name}' already exists")
            return True
    
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
    
    def bulk_index(self, datas):
        """
        Index multiple documents in bulk.
        
        Args:
            datas (list): List of datas to embed and index
            
        Returns:
            list: List of document IDs
        """
        self.model = SentenceTransformer(model_name)
        doc_ids = []
        
        # Process in batches to avoid memory issues with large datasets
        batch_size = 100
        for i in tqdm(range(0, len(datas), batch_size), desc="Indexing batches"):
            batch_datas = datas[i:i + batch_size]
            
            # Generate embeddings for the batch
            batch_texts = [f"{data['title']} {data['abstract']}" for data in batch_datas]
            embeddings = self.model.encode(batch_texts)
            
            # Prepare bulk indexing operations
            operations = []
            for j, (data, embedding) in enumerate(zip(batch_datas, embeddings)):
                # Add index action
                action = {"index": {"_index": self.index_name}}
                if data["paperId"]:
                    action["index"]["_id"] = data["paperId"]
                
                operations.append(action)

                metadata = {
                    "paperId": data["paperId"],
                    "title": data["title"],
                    "abstract": data["abstract"],
                    "url": data["url"],
                    "year": data["year"],
                    "authors": data["authors"],
                }
                
                # Add document source
                doc = {
                    "text": data["title"],
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

# Example usage
if __name__ == "__main__":
    with open("data.json") as f:
        datas = json.load(f)
    datas = datas["data"]
    
    # Initialize the indexer
    indexer = TextEmbeddingIndexer()
    
    # Create the index
    is_exist = indexer.create_index()
    if not is_exist:
        # Index the semantic scholar datas
        print("Indexing semantic scholar datas...")
        doc_ids = indexer.bulk_index(datas)
    
        print(f"Indexed {len(doc_ids)} documents")
    else:
        print("Index already exists")
