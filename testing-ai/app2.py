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

    # --- Bookkeeping Field Extraction Using AI ---
    st.subheader("Extracted Bookkeeping Fields")

    # Prepare the document content for the prompt
    document_text = " ".join([doc.page_content for doc in documents])

    # Create the LLM prompt for bookkeeping field extraction
    prompt_text = f"""
    You are an AI financial assistant. Please analyze the following document and extract the relevant bookkeeping fields. Look for the following types of information:
    - **Transaction Amounts**: Identifiable monetary values.
    - **Dates**: Transaction dates in standard formats.
    - **Descriptions**: Descriptive text explaining each transaction.
    - **References**: Any alphanumeric codes that could identify or reference transactions (e.g., invoice numbers, transaction IDs).

    Only extract relevant fields that fit these categories and ignore non-relevant data. Here is the document content:

    {document_text}

    Please return a structured list of the following fields: 
    - Date
    - Transaction Amount
    - Description
    - Reference
    """

    # Send the prompt to the model for processing
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI financial assistant."},
            {"role": "user", "content": prompt_text}
        ]
    )

    # Parse and display the extracted bookkeeping fields
    extracted_fields = response.choices[0].message.content.strip()
    st.write("AI Response:")
    st.write(extracted_fields)

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

    # Clean up temporary file
    os.remove(file_path)

# --- Footer ---
st.divider()
st.markdown("Source code: [Github](https://github.com/your-repository-link)")
