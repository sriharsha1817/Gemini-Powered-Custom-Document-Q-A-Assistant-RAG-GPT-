# 🤖 RAG-GPT: Gemini-Powered Document Q&A Assistant

RAG-GPT is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents (PDF, DOCX, TXT), ask questions, and receive smart answers powered by Google's Gemini API. It uses TF-IDF + cosine similarity for document retrieval and a custom Streamlit UI for interaction.

---

## 🔍 Features

- 📄 Upload and process multiple document types (PDF, DOCX, TXT)
- 🔍 Automatic text extraction and chunking
- 🧠 RAG-based question answering using TF-IDF and cosine similarity
- 🤖 Context-aware responses using Gemini Pro LLM
- 🧾 Source tracking and citation for transparency
- 💬 Persistent chat history for conversation context
- 🌐 Web interface using Streamlit

---

## 🖥️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Gemini API
- **NLP**: TF-IDF (scikit-learn), cosine similarity
- **Document Parsing**: PyPDF2, python-docx

---
📁 Project Structure

    RAG-GPT/
    |
    │── app.py                # Main Streamlit app
    ├── requirements.txt          # Required Python libraries
    └── README.md                 # This file


## ⚙️ Setup Instructions

 1. Clone the repository
   in bash

        git clone https://github.com/sriharsha1817/Gemini-Powered-Custom-Document-Q-A-Assistant-RAG-GPT-.git
        cd rag-gpt
2. Install dependencies

        pip install -r requirements.txt
3.Get Gemini API Key:
 Create a free API key
 Enter it in the sidebar when running the app

    Visit https://makersuite.google.com/app/apikey
4. Run the app

       streamlit run app.py

🧠 Example Use Cases:

📚 Ask questions about course materials

🧾 Analyze legal documents or policies

💼 Summarize resumes or reports

## 🔗 ✨ Live Demo:

🚀 Try the app here: [https://rag-gpt.streamlit.app](https://exwyhkjn4dndeitm5yjwt4.streamlit.app/)


