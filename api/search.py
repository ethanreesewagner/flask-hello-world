from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
def search(query,id):
    index = pc.Index("testingvectors")
    query_vector = client.embeddings.create(input=query,
    model="text-embedding-3-small", dimensions=512)
    results = index.query(vector=list(query_vector.data[0].embedding), top_k=5, include_metadata=True, namespace=id)
    return results

