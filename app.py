import streamlit as st
from ingest import load_documents, split_documents
from rag_chatbot import build_vector_store, get_qa_chain

@st.cache_resource
def load_bot():
    docs = load_documents()
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks)
    return get_qa_chain(vector_store)

st.title("Angel One Support Chatbot")
query = st.text_input("Ask a question...")

qa_chain = load_bot()

if query:
    result = qa_chain.invoke({"query": query})
    answer = result['result']
    if "I don't know" in answer or not answer.strip():
        st.info("I don't know.")
    else:
        st.success(answer)
