from flask import Flask, request
from flask.cli import routes_command
from flask.typing import RouteCallable
from langchain_community.document_loaders.pdf import PyPDFLoader
import tempfile
import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from openai import OpenAI
from .agent_functions import process_user_input
from flask_cors import CORS
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/agent', methods=['POST'])
def send_to_agent():
    try:
        logger.info("Received request to /agent endpoint")
        data = request.get_json()
        if not data:
            logger.warning("No JSON data received in /agent request")
            return "No data provided", 400
        
        message = data.get("message")
        conversation_id = data.get("id")
        
        if not message:
            logger.warning("No message provided in /agent request")
            return "No message provided", 400
        
        if not conversation_id:
            logger.warning("No conversation ID provided in /agent request")
            return "No conversation ID provided", 400
        
        logger.info(f"Processing agent request - conversation_id: {conversation_id}, message length: {len(message)}")
        result = process_user_input(message, conversation_id)
        logger.info(f"Agent request completed successfully for conversation_id: {conversation_id}")
        return result
    except Exception as e:
        logger.error(f"Error in /agent endpoint: {str(e)}", exc_info=True)
        return f"Error processing request: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Starting upload_file function")
        if "file" not in request.files:
            logger.warning("No file provided in upload request")
            return "No file provided", 400
        
        file = request.files["file"]
        logger.info(f"File received: {file.filename}, content_type: {file.content_type}")
        if "id" not in request.form:
            logger.warning("No conversation ID provided in upload request")
            return "No conversation ID provided", 400
        
        conversation_id = request.form["id"]
        logger.info(f"Upload request for conversation_id: {conversation_id}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            file.save(temp_pdf_file.name)
            temp_pdf_path = temp_pdf_file.name
        logger.info(f"File saved to temporary path: {temp_pdf_path}")
        # Initialize PyPDFLoader with the path to the temporary file
        loader = PyPDFLoader(temp_pdf_path)
        # Properly load the document using the loader
        docs = loader.load()
        logger.info(f"PDF loaded. Number of documents (pages): {len(docs)}")
        # WARNING: The original code returned only the first page. This has been fixed to process all pages.

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("testingvectors")
        logger.info("OpenAI client and Pinecone index initialized")

        all_vectors = [] # To accumulate vectors from all pages

        for page_num, doc in enumerate(docs):
            logger.info(f"Processing page {page_num + 1}/{len(docs)} for conversation_id: {conversation_id}")
            # Process each page individually
            chunks = text_splitter.split_documents([doc]) # Split a single page
            texts = [chunk.page_content for chunk in chunks]
            logger.debug(f"Page {page_num + 1} split into {len(chunks)} chunks.")

            if not texts:
                logger.warning(f"No text found for page {page_num + 1}, skipping.")
                continue

            logger.info(f"Generating embeddings for {len(texts)} text chunks from page {page_num + 1}")
            response = client.embeddings.create(input=texts, model="text-embedding-3-small", dimensions=512)
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Embeddings generated for page {page_num + 1}")

            vectors_for_page = []
            for i, embedding in enumerate(embeddings):
                vectors_for_page.append({
                    "id": f"chunk_{page_num}_{i}", # Unique ID for each chunk across pages
                    "values": embedding,
                    "metadata": {"title": temp_pdf_path, "text": texts[i], "page": chunks[i].metadata["page"]}
                })
            all_vectors.extend(vectors_for_page) # Add to the list of all vectors
            
            logger.info(f"Upserting {len(vectors_for_page)} vectors for page {page_num + 1} to Pinecone namespace: {conversation_id}")
            upsert_response = index.upsert(
                vectors=vectors_for_page,
                namespace=conversation_id
            )
            logger.debug(f"Upsert complete for page {page_num + 1}. Response: {upsert_response}")
        
        # Clean up the temporary file
        logger.info("Cleaning up temporary file")
        os.unlink(temp_pdf_path)
        logger.debug("Temporary file cleaned up")
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
            logger.info(f"Successfully processed and upserted {len(all_vectors)} vectors for conversation_id: {conversation_id}")
            # Instead of returning a descriptor, return the actual vectors data that was upserted
            # Ensure all objects are JSON serializable
            serializable_vectors = make_serializable(all_vectors)
            return temp_pdf_path
        else:
            logger.warning(f"No content found after processing for conversation_id: {conversation_id}")
            return "No content found."
    except Exception as e:
        logger.error(f"An error occurred in upload_file: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}"
