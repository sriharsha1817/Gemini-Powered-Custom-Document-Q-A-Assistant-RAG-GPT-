import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io
import os
from typing import List, Dict, Any

# Configure page
st.set_page_config(
    page_title="Custom RAG GPT",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'doc_vectors' not in st.session_state:
    st.session_state.doc_vectors = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class DocumentProcessor:
    """Handle document processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.documents = []
        self.doc_vectors = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the knowledge base"""
        self.documents.extend(documents)
        
        # Extract text from all documents
        all_text = [doc['content'] for doc in self.documents]
        
        # Vectorize documents
        self.doc_vectors = self.vectorizer.fit_transform(all_text)
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for a query"""
        if not self.documents or self.doc_vectors is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                relevant_docs.append({
                    'content': self.documents[idx]['content'],
                    'source': self.documents[idx]['source'],
                    'similarity': similarities[idx]
                })
        
        return relevant_docs

def setup_gemini_api():
    """Setup Gemini API configuration"""
    api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def generate_response(query: str, context: str, chat_history: List[Dict]) -> str:
    """Generate response using Gemini API with RAG context"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Build conversation history
        history_text = ""
        for msg in chat_history[-5:]:  # Last 5 messages for context
            history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        prompt = f"""
        You are a helpful AI assistant. Use the following context from documents to answer the user's question.
        If the answer is not in the context, say so and provide a general response.
        
        Context from documents:
        {context}
        
        Previous conversation:
        {history_text}
        
        Current question: {query}
        
        Please provide a helpful and accurate response:
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("🤖 Custom RAG GPT")
    st.markdown("Upload documents and chat with your custom AI assistant powered by Gemini API")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key setup
    if not setup_gemini_api():
        st.sidebar.warning("Please enter your Gemini API key to continue")
        st.info("Get your free Gemini API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Document upload section
    st.sidebar.header("Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    # Process uploaded documents
    if uploaded_files:
        with st.sidebar:
            with st.spinner("Processing documents..."):
                new_docs = []
                for file in uploaded_files:
                    if file.type == "application/pdf":
                        text = DocumentProcessor.extract_text_from_pdf(file)
                    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = DocumentProcessor.extract_text_from_docx(file)
                    else:  # txt file
                        text = str(file.read(), "utf-8")
                    
                    if text:
                        chunks = DocumentProcessor.chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            new_docs.append({
                                'content': chunk,
                                'source': f"{file.name} (chunk {i+1})"
                            })
                
                if new_docs:
                    st.session_state.documents.extend(new_docs)
                    rag.add_documents(st.session_state.documents)
                    st.session_state.vectorizer = rag.vectorizer
                    st.session_state.doc_vectors = rag.doc_vectors
                    st.success(f"Processed {len(new_docs)} document chunks")
    
    # Load existing documents if available
    if st.session_state.documents and st.session_state.vectorizer:
        rag.documents = st.session_state.documents
        rag.vectorizer = st.session_state.vectorizer
        rag.doc_vectors = st.session_state.doc_vectors
    
    # Document status
    st.sidebar.markdown(f"**Documents loaded:** {len(st.session_state.documents)}")
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for i, msg in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {msg['user'][:50]}..."):
                st.write(f"**User:** {msg['user']}")
                st.write(f"**Assistant:** {msg['assistant']}")
                if 'sources' in msg:
                    st.write("**Sources:**")
                    for source in msg['sources']:
                        st.write(f"- {source}")
    
    # Chat input
    with st.form("chat_form"):
        user_input = st.text_area("Ask a question:", height=100)
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        with st.spinner("Generating response..."):
            # Retrieve relevant documents
            relevant_docs = rag.retrieve_relevant_docs(user_input)
            
            # Build context from relevant documents
            context = ""
            sources = []
            for doc in relevant_docs:
                context += f"Source: {doc['source']}\nContent: {doc['content']}\n\n"
                sources.append(doc['source'])
            
            # Generate response
            response = generate_response(user_input, context, st.session_state.chat_history)
            
            # Add to chat history
            chat_entry = {
                'user': user_input,
                'assistant': response,
                'sources': sources
            }
            st.session_state.chat_history.append(chat_entry)
            
            # Display current response
            st.subheader("Response")
            st.write(response)
            
            if sources:
                st.subheader("Sources")
                for source in sources:
                    st.write(f"- {source}")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()