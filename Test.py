import streamlit as st
import numpy as np
from pypdf import PdfReader
from groq import Groq


# -----------------------------
# GROQ API KEY
# -----------------------------

client = Groq(api_key="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")


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")


# -----------------------------
# LOAD PDF
# -----------------------------

@st.cache_data
def load_pdf():

    reader = PdfReader("Report.pdf")

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


# -----------------------------
# TEXT CHUNKING
# -----------------------------

def chunk_text(text, chunk_size=1000, overlap=200):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# -----------------------------
# SIMPLE NUMPY EMBEDDINGS
# -----------------------------

def embed_text(text, dim=384):

    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(dim)


# -----------------------------
# CREATE VECTOR DATABASE
# -----------------------------

@st.cache_resource
def create_embeddings(chunks):

    embeddings = [embed_text(chunk) for chunk in chunks]

    return np.array(embeddings)


# -----------------------------
# COSINE SIMILARITY
# -----------------------------

def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# RETRIEVAL
# -----------------------------

def retrieve(query, chunks, embeddings, top_k=3):

    query_embedding = embed_text(query)

    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in embeddings
    ]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [chunks[i] for i in top_indices]


# -----------------------------
# STREAMING RESPONSE
# -----------------------------

def stream_answer(prompt):

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    full_response = ""

    for chunk in stream:

        if chunk.choices[0].delta.content is not None:

            token = chunk.choices[0].delta.content
            full_response += token

            yield full_response


# -----------------------------
# LOAD DATA
# -----------------------------

text = load_pdf()
chunks = chunk_text(text)
embeddings = create_embeddings(chunks)


# -----------------------------
# CHAT MEMORY
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# USER INPUT
# -----------------------------

query = st.chat_input("Ask something about the PDF...")


if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)


    # Retrieve relevant chunks

    relevant_chunks = retrieve(query, chunks, embeddings)

    context = "\n\n".join(relevant_chunks)[:4000]


    # Chat history

    history = st.session_state.messages[-4:]

    history_text = ""

    for h in history:
        history_text += f"{h['role']}: {h['content']}\n"


    prompt = f"""
You are a helpful assistant answering questions about a document.

Conversation History:
{history_text}

Context:
{context}

User Question:
{query}
"""


    # Assistant response

    with st.chat_message("assistant"):

        placeholder = st.empty()

        final_answer = ""

        for partial in stream_answer(prompt):

            final_answer = partial
            placeholder.markdown(final_answer + "▌")

        placeholder.markdown(final_answer)


    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )