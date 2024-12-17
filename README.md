# Overview
# Chat-with-pdfs-using-RAG-pipelines
A Retrieval-Augmented Generation (RAG) pipeline that enables users to query PDF documents and receive accurate, context-rich responses. The system uses embeddings, FAISS for similarity search, and a pre-trained embedding model for efficient text retrieval.

# Features
PDF Text Extraction: Extracts text content from PDF files.

Embeddings Generation: Converts extracted text into embeddings using a pre-trained model.

Vector Database: Stores embeddings in a FAISS index for efficient similarity search.

Query Handling: Accepts user queries, searches for relevant text, and returns context-rich responses.

Fast and Scalable: Processes large PDFs and handles queries quickly.

# Technologies Used
Programming Language: Python

Libraries:
PyPDF2: Extract text from PDFs.

sentence-transformers: Generate vector embeddings.

FAISS: Fast similarity search on embeddings.

numpy: For numerical operations and embedding handling.




