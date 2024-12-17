import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to extract text from the uploaded PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=2,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get TF-IDF embeddings for text chunks
def get_tfidf_embeddings(text_chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    embeddings = vectorizer.fit_transform(text_chunks)
    return embeddings, vectorizer

# Function to search for the most similar chunk given a query
def search_query(query, vectorizer, embeddings_array, text_chunks):
    query_embedding = vectorizer.transform([query]).toarray()  # Convert query to vector
    cos_sim = cosine_similarity(query_embedding, embeddings_array)  # Compute similarity
    most_similar_idx = np.argmax(cos_sim)  # Get the index of the most similar chunk
    return text_chunks[most_similar_idx]  # Return the most similar chunk

# Main Streamlit app
def main():
    load_dotenv()  # Load environment variables from .env file (optional)
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.header("Chat with your PDF :page_facing_up:")
    
    # Text input for the query
    query = st.text_input("Ask any question related to your PDF:")

    with st.sidebar:
        st.subheader("Uploaded Documents")
        # File uploader for multiple PDFs
        pdf_docs = st.file_uploader("Upload your PDF here:", accept_multiple_files=True)

        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Step 1: Extract text from the uploaded PDFs
                    raw_text = get_pdf_text(pdf_docs)

                    # Step 2: Split the extracted text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Step 3: Get TF-IDF embeddings for the text chunks
                    embeddings, vectorizer = get_tfidf_embeddings(text_chunks)

                    # Convert embeddings to a dense array for cosine similarity
                    embeddings_array = embeddings.toarray()

                    # Step 4: Perform similarity search if a query is entered
                    if query:
                        most_similar_chunk = search_query(query, vectorizer, embeddings_array, text_chunks)
                        st.write("Most similar chunk:", most_similar_chunk)
            else:
                st.warning("Please upload a PDF document.")

if __name__ == "_main_":
    main()
