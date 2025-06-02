import os
import faiss
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
import joblib
import streamlit as st
from openai import OpenAI, APIConnectionError, AuthenticationError
from sentence_transformers import SentenceTransformer

# Directory where PDFs will be stored and processed
PDF_DIR = r'C:\Users\Hitesh\Downloads\Class 9th.zip'
print("FITZ MODULE:", fitz.__file__)  # Debugging module path

# Extracts text from each page of the PDF
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return " ".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Failed to read {pdf_path}: {e}")
        return ""

# Splits large text into manageable chunks (based on max token count)
def chunk_text(text, max_tokens=500):
    paragraphs = text.split('\n\n')
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk.split()) + len(para.split()) <= max_tokens:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Queries Groq's LLM with prompt and returns the result
def query_groq(prompt):
    try:
        client = OpenAI(
            api_key='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',
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except AuthenticationError:
        return "❌ Authentication failed: Invalid API key."
    except APIConnectionError as e:
        return f"❌ Connection error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# Extracts and embeds text chunks from uploaded PDFs and stores them with FAISS index
def prepare_embeddings():
    all_chunks = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(PDF_DIR, filename)
            text = extract_text_from_pdf(full_path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No valid chunks extracted from PDFs.")
        return

    st.write(f"✅ Extracted {len(all_chunks)} text chunks.")

    # Load embedding model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    # Create FAISS index for fast vector search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save models and data
    joblib.dump(embedder, "embedder.joblib")
    joblib.dump(all_chunks, "text_chunks.joblib")
    np.save("embeddings.npy", embeddings)
    faiss.write_index(index, "faiss_index.index")

    st.success("Embeddings created and saved!")

# Loads saved components (model, index, chunks)
def load_components():
    embedder = joblib.load("embedder.joblib")
    chunks = joblib.load("text_chunks.joblib")
    index = faiss.read_index("faiss_index.index")
    return embedder, chunks, index

# Searches top matching chunks and sends a contextual prompt to the LLM for a response
def search_and_answer(query, embedder, chunks, index, user_type, weak_subjects, language):
    try:
        query_vec = embedder.encode([query], convert_to_numpy=True)
        D, I = index.search(query_vec, k=3)
        retrieved = [chunks[i] for i in I[0] if i != -1]

        # Limit context length for LLM input
        max_chars = 16000
        context = ""
        for chunk in retrieved:
            if len(context) + len(chunk) < max_chars:
                context += "\n\n" + chunk
            else:
                break

        # Personalize based on user input
        personalization = f"You are answering a question asked by a {user_type.lower()}."
        if weak_subjects:
            personalization += f" They struggle with: {', '.join(weak_subjects)}."

        lang_instruction = "Respond in English." if language == "English" else "उत्तर हिंदी में दें।"

        # Final prompt construction
        prompt = f"""{personalization}
{lang_instruction}

Context:
{context}

Question: {query}
Answer:"""

        return query_groq(prompt)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------
# Streamlit Web Interface
# ---------------------------
st.set_page_config(page_title="📚 Class 9 Chatbot", layout="centered")
st.title("📚 Class 9 Chatbot")
st.markdown("Ask questions from your uploaded Class 9 textbook PDFs.")

# Ensure PDF directory exists
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)

# Upload section
uploaded_files = st.file_uploader("Upload Class 9 PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join(PDF_DIR, file.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file.read())
    st.success("📄 PDFs uploaded!")

# Build or refresh knowledge base
if st.button("🔁 Build/Refresh Knowledge Base"):
    prepare_embeddings()

# User customization options
user_type = st.selectbox("👤 Select user type:", ["Student", "Teacher", "Parent", "Other"])
weak_subjects = st.multiselect("❗ Select weak subject areas (optional):", ["Math", "Science", "English", "Social Studies", "Hindi"])
language = st.selectbox("🌐 Select language for response:", ["English", "Hindi"])

# Load components if all required files are present
required_files = ["faiss_index.index", "embedder.joblib", "text_chunks.joblib"]
if all(os.path.exists(f) for f in required_files):
    embedder, chunks, index = load_components()

    # Query input and response section
    user_query = st.text_input("💬 Ask a question:")
    if user_query:
        with st.spinner("Thinking..."):
            result = search_and_answer(user_query, embedder, chunks, index, user_type, weak_subjects, language)
        st.success(result)
else:
    st.warning("Please upload PDFs and click 'Build Knowledge Base' to enable chat.")
