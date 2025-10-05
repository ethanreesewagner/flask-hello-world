from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("testingvectors")
def search(query,file_path):
    # Add a filter to only return results where the metadata "title" matches a specific value.
    # For demonstration, we'll prompt the user for the title to filter by.
    query_vector = client.embeddings.create(input=query,
    model="text-embedding-3-small", dimensions=512)
    query_filter = {
    "title": {"$eq": file_path}
    }
    results = index.query(vector=list(query_vector.data[0].embedding), top_k=5, include_metadata=True, namespace="example-namespace", filter=query_filter)
    return results

