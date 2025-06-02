# Study-ChatBot

📚 Class 9 PDF-based Chatbot (Local Setup)
This is a local chatbot app built with Streamlit, powered by Groq's LLaMA-3 model, and supports Class 9 students, teachers, and parents in querying PDF textbooks.
 Key Features
Upload Class 9 textbook PDFs


Automatically extract and chunk content


Use sentence embeddings and FAISS for fast retrieval


Ask subject-related questions and receive intelligent answers


Supports English and Hindi responses


Personalized answers based on user type and weak subjects


📂 Project Folder Structure
📁 class9_chatbot/
│
├── main.py                  # Main Streamlit application file
├── requirements.txt        # Python dependency list
├── faiss_index.index       # Auto-generated FAISS index file (after embedding)
├── embedder.joblib         # Auto-generated embedding model
├── text_chunks.joblib      # Auto-generated text chunks
├── embeddings.npy          # Auto-generated embedding matrix
└── 📁 test/Class9th/        # Folder to upload and store PDFs


✅ Prerequisites
Python 3.8 or higher installed


Internet connection (for downloading models and calling Groq API)


Your Groq API key


 Setup Instructions
1. Create a New Folder
Create a folder anywhere (e.g., D:\Projects\class9_chatbot) and place your app.py file there.
2. Install Python Virtual Environment (Recommended)
cd D:\Projects\class9_chatbot
python -m venv venv
venv\Scripts\activate    # On Windows

3. Install Required Python Packages
Create a file requirements.txt in the same folder with the following content:
streamlit
openai
faiss-cpu
joblib
numpy
PyMuPDF
sentence-transformers

Then run:
pip install -r requirements.txt

4. Set PDF Directory
In your app.py, make sure this line points to a valid folder:
PDF_DIR = r'C:\Users\parih\Downloads\test\Class9th'

Create that folder manually or change the path to something like:
PDF_DIR = r'D:\Projects\class9_chatbot\test\Class9th'

5. Add Your Groq API Key
Replace the key in the query_groq() function:
client = OpenAI(
    api_key='your_groq_api_key_here',
    base_url="https://api.groq.com/openai/v1"
)

How to Run the App
From the project directory, run:
streamlit run main.py

This will open the chatbot in your browser at:
http://localhost:8501

Usage Flow
Upload PDFs using the Streamlit interface.


Click "🔁 Build/Refresh Knowledge Base".


Select user type, weak subjects, and preferred language.


Ask your question in the input box.


View the response generated using your document content.


📝 Notes
PDF uploads are saved locally under the folder you configured in PDF_DIR.


Embeddings are cached — no need to rebuild unless PDFs change.


LLaMA model access uses Groq's fast inference API. Make sure your API key is valid.




