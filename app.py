import streamlit as st
from ingest import load_documents, split_documents
from rag_chatbot import build_vector_store, get_qa_chain
import time

@st.cache_resource
def load_bot():
    with st.spinner('Initializing the chatbot... This may take a few minutes.'):
        docs = load_documents()
        chunks = split_documents(docs)
        vector_store = build_vector_store(chunks)
        return get_qa_chain(vector_store)

st.title("Angel One Support Chatbot")
st.markdown("""
    <style>
    .stSpinner > div {
        text-align: center;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
query = st.chat_input("Ask a question...")

if query:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Get bot response with loading indicator
    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            qa_chain = load_bot()
            result = qa_chain.invoke({"query": query})
            answer = result['result']
            
            if "I don't know" in answer or not answer.strip():
                st.info("I don't know.")
            else:
                st.success(answer)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
