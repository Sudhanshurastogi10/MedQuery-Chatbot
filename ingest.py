from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import logging

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
BATCH_SIZE = 1000  # Adjust batch size based on available memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_db():
    logger.info("Loading documents from directory...")
    start_time = time.time()

    try:
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return

    if not documents:
        logger.info("No documents found.")
        return

    logger.info(f"Loaded {len(documents)} documents. Time taken: {time.time() - start_time:.2f} seconds")

    for i, doc in enumerate(documents):
        logger.info(f"Document {i + 1}/{len(documents)}: {doc.metadata['source']}")

    logger.info("Splitting documents into chunks...")
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(texts)} chunks. Time taken: {time.time() - start_time:.2f} seconds")

    logger.info("Generating embeddings for the chunks...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    logger.info(f"Embeddings model loaded. Time taken: {time.time() - start_time:.2f} seconds")

    logger.info("Creating FAISS vector store in batches...")
    start_time = time.time()
    db = None

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        if db is None:
            db = FAISS.from_documents(batch_texts, embeddings)
        else:
            db.add_documents(batch_texts)

        logger.info(f"Processed batch {i // BATCH_SIZE + 1}/{len(texts) // BATCH_SIZE + 1}")

    logger.info(f"FAISS vector store created. Time taken: {time.time() - start_time:.2f} seconds")

    logger.info("Saving FAISS vector store locally...")
    start_time = time.time()
    db.save_local(DB_FAISS_PATH)
    logger.info(f"Vector store saved successfully. Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data directory '{DATA_PATH}' does not exist.")
    else:
        create_vector_db()
