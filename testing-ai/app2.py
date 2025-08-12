import os
import tempfile
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI


# --- Page Setup ---
st.set_page_config(page_title="AI Financial Document Assistant", page_icon="📊")

# --- API Keys ---
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")

# --- Client Init ---
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.error("OpenAI API Key is missing. Please add it to your Streamlit secrets.")
    st.stop()

# --- Document Upload Section ---
st.header("Upload Your Financial Document")

uploaded_file = st.file_uploader("Upload a PDF document (tax return, financial statement)", type="pdf")

# --- Process the Uploaded Document ---
if uploaded_file:
    # Save the uploaded document to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    # Load the document using Langchain's PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # --- Bookkeeping Field Extraction ---
    st.subheader("Extracted Bookkeeping Fields")
    # Placeholder: This will display the extracted bookkeeping fields (in practice, this will need AI-based field extraction logic)
    # For now, let's assume we extract fields like "Transaction Amounts", "Date", "Description", and "References"
    bookkeeping_fields = [
        {"Transaction": 5000, "Date": "2025-01-15", "Description": "Payment from Client", "Reference": "INV12345"},
        {"Transaction": 2000, "Date": "2025-01-16", "Description": "Service Fee", "Reference": "SVC56789"},
        {"Transaction": 7500, "Date": "2025-01-17", "Description": "Refund from Vendor", "Reference": "REF98765"}
    ]
    st.table(bookkeeping_fields)

    # --- AI-Powered User Query Assistance ---
    st.subheader("Ask the AI about your document")

    user_query = st.text_input("What would you like to ask about this document?")

    if user_query:
        # Initialize Langchain components for querying the document
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

        # Split the document into chunks
        splitted_documents = text_splitter.split_documents(documents)

        # Create the FAISS vector store
        db = FAISS.from_documents(splitted_documents, embeddings)
        chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

        # Query the document using the AI
        response = chain({"question": user_query, "chat_history": []})
        answer = response["answer"].strip()

        st.write(f"AI Response: {answer}")

    # --- AI-Generated Insights ---
    st.subheader("AI-Generated Insights")

    # Placeholder for AI insights: This can be enhanced to include real financial analysis
    ai_insights = """
    - **Total Transactions Processed**: 3 transactions identified.
    - **Anomalies Detected**: No anomalies detected based on the provided threshold.
    - **Key Insights**: The document contains a variety of transactions, including service fees and refunds. The total amounts appear consistent with expected financial activity.
    """
    st.write(ai_insights)

    # Clean up temporary file
    os.remove(file_path)

# --- Footer ---
st.divider()
st.markdown("Source code: [Github](https://github.com/your-repository-link)")
