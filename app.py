import streamlit as st
import os
import tempfile
from rag_utils import load_and_process_file, setup_vectorstore, get_rag_chain, query_rag

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 Chat with your Document")

# Sidebar for configuration and file upload
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.session_state.vectorstore = None
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "last_performance" not in st.session_state:
    st.session_state.last_performance = None

# Handle file upload
if uploaded_file is not None and st.session_state.rag_chain is None:
    with st.spinner("Processing file..."):
        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Process file
            splits = load_and_process_file(tmp_file_path)
            vectorstore = setup_vectorstore(splits)
            st.session_state.rag_chain = get_rag_chain(vectorstore)
            
            # Cleanup
            os.remove(tmp_file_path)
            st.success("File processed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    if st.session_state.rag_chain is None:
        st.warning("Please upload a document first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, latency, context = query_rag(
                        st.session_state.rag_chain, 
                        prompt, 
                        session_id="user_session"
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Store performance metrics
                    st.session_state.last_performance = {
                        "latency": latency,
                        "context": context
                    }
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Performance Metrics
if st.session_state.last_performance:
    with st.expander("Show RAG Performance"):
        st.metric("Retrieval & Generation Latency", f"{st.session_state.last_performance['latency']:.4f} seconds")
        st.subheader("Retrieved Context")
        for i, doc in enumerate(st.session_state.last_performance["context"]):
            st.markdown(f"**Source {i+1}:**")
            st.text(doc.page_content)
            st.divider()
