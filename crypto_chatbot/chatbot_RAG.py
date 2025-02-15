import os
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ProcessPoolExecutor
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def extract_columns_sequentially(pdf_path):
    all_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                width = page.width
                left_bbox = (0, 0, width / 2, page.height)
                right_bbox = (width / 2, 0, width, page.height)
                left_text = page.within_bbox(left_bbox).extract_text() or ""
                right_text = page.within_bbox(right_bbox).extract_text() or ""
                all_text.append(left_text)
                all_text.append(right_text)
    except Exception as e:
        logger.error(f"Error extracting columns from {pdf_path}: {e}")
    return "\n".join(all_text)

def process_pdf(pdf_path, is_two_column=True, num_title_lines=5):
    try:
        if is_two_column:
            text = extract_columns_sequentially(pdf_path)
        else:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata or {}
    except Exception as e:
        logger.error(f"Error reading metadata from {pdf_path}: {e}")
        metadata = {}

    author = metadata.get("Author", "Unknown")
    lines = text.splitlines()
    title = " ".join(lines[:num_title_lines]) if lines else "Untitled"
    meta = {"author": author, "title": title, "source": pdf_path}
    return {"text": text, "metadata": meta}

def chunk_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def build_documents_from_folder(folder_path, is_two_column=True):
    docs = []
    for filename in os.listdir(folder_path):
       
        pdf_path = os.path.join(folder_path, filename)
        try:
            result = process_pdf(pdf_path, is_two_column=is_two_column)
            if not result["text"]:
                logger.warning(f"No text extracted from {pdf_path}")
                continue
            chunks = chunk_text(result["text"])
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata=result["metadata"])
                docs.append(doc)
            logger.info(f"Processed {pdf_path}: {len(chunks)} chunks created.")
        except Exception as e:
            logger.error(f"Error processing file {pdf_path}: {e}")
    return docs

def build_vector_store(documents, embedding_model):
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store

def process_folders(folders, is_two_column_flags, embedding_model):
    all_docs = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for folder, is_two_column in zip(folders, is_two_column_flags):
            futures.append(executor.submit(build_documents_from_folder, folder, is_two_column))
        for future in futures:
            try:
                docs = future.result()
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Error in process pool: {e}")
    vector_store = build_vector_store(all_docs, embedding_model)
    return vector_store

if __name__ == "__main__":
    folders = [os.path.join("pdfs", "half_page"), "pdfs"]
    is_two_column_flags = [True, False]
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    try:
        vector_store = process_folders(folders, is_two_column_flags, embedding_model)
        vector_store.save_local("vector_store")
        logger.info("Vector store built successfully with {} documents.".format(
            len(vector_store.index_to_docstore_id)
        ))
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
