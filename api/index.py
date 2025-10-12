from flask import Flask, request
from flask.cli import routes_command
from flask.typing import RouteCallable
from langchain_community.document_loaders.pdf import PyPDFLoader
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from openai import OpenAI
from .agent_functions import process_user_input

load_dotenv()
app = Flask(__name__)

@app.route('/agent', methods=['POST'])
def send_to_agent():
    data = request.get_json()
    message=data["message"]
    path=data["path"]
    return process_user_input(message+f" The path is {path}")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Starting upload_file function")
        file = request.files["file"]
        print(f"File received: {file.filename}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            file.save(temp_pdf_file.name)
            temp_pdf_path = temp_pdf_file.name
        print(f"File saved to temporary path: {temp_pdf_path}")
        # Initialize PyPDFLoader with the path to the temporary file
        loader = PyPDFLoader(temp_pdf_path)
        # Properly load the document using the loader
        docs = loader.load()
        print(f"PDF loaded. Number of documents (pages): {len(docs)}")
        # WARNING: The original code returned only the first page. This has been fixed to process all pages.

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("testingvectors")
        print("OpenAI client and Pinecone index initialized")

        all_vectors = [] # To accumulate vectors from all pages

        for page_num, doc in enumerate(docs):
            print(f"Processing page {page_num + 1}/{len(docs)}")
            # Process each page individually
            chunks = text_splitter.split_documents([doc]) # Split a single page
            texts = [chunk.page_content for chunk in chunks]
            print(f"Page {page_num + 1} split into {len(chunks)} chunks.")

            if not texts:
                print(f"No text found for page {page_num + 1}, skipping.")
                continue

            print(f"Generating embeddings for {len(texts)} text chunks from page {page_num + 1}")
            response = client.embeddings.create(input=texts, model="text-embedding-3-small", dimensions=512)
            embeddings = [data.embedding for data in response.data]
            print(f"Embeddings generated for page {page_num + 1}")

            vectors_for_page = []
            for i, embedding in enumerate(embeddings):
                vectors_for_page.append({
                    "id": f"chunk_{page_num}_{i}", # Unique ID for each chunk across pages
                    "values": embedding,
                    "metadata": {"title": temp_pdf_path, "text": texts[i], "page": chunks[i].metadata["page"]}
                })
            all_vectors.extend(vectors_for_page) # Add to the list of all vectors
            
            print(f"Upserting {len(vectors_for_page)} vectors for page {page_num + 1} to Pinecone")
            upsert_response = index.upsert(
                vectors=vectors_for_page,
                namespace="example-namespace"
            )
            print(f"Upsert complete for page {page_num + 1}. Response: {upsert_response}")
        
        # Clean up the temporary file
        print("Cleaning up temporary file")
        os.unlink(temp_pdf_path)
        print("Temporary file cleaned up")
        # Convert upsert_response to dict if possible for serialization
        # Ensure the response is JSON serializable
        from flask import jsonify

        def make_serializable(obj):
            # Recursively convert objects to serializable types
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, "to_dict"):
                return make_serializable(obj.to_dict())
            elif hasattr(obj, "__dict__"):
                return make_serializable(vars(obj))
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)

        if all_vectors: # Check all_vectors instead of docs
            print("Successfully processed and upserted vectors. Returning success.")
            # Instead of returning a descriptor, return the actual vectors data that was upserted
            # Ensure all objects are JSON serializable
            serializable_vectors = make_serializable(all_vectors)
            return temp_pdf_path
        else:
            print("No content found after processing. Returning no content message.")
            return "No content found."
    except Exception as e:
        print(f"An error occurred in upload_file: {e}")
        return f"An error occurred: {e}"
