import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# --- Page Setup ---
st.set_page_config(page_title="AI + OCR-Powered Financial Data Automation Assistant", page_icon="📊")
st.title("AI Financial Data Assistant")

# --- Provider Selection ---
provider = st.selectbox("🧠 Choose your AI model:", ["OpenAI"])  #, "Google Gemini"

# --- API Keys ---
openai_api_key = st.secrets.get("openai_api_key")
gemini_api_key = st.secrets.get("gemini_api")

# --- Client Init ---
if provider == "OpenAI":
    if not openai_api_key:
        st.error("Missing OpenAI API key.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)
elif provider == "Google Gemini":
    if not gemini_api_key:
        st.error("Missing Gemini API key.")
        st.stop()
    genai.configure(api_key=gemini_api_key)
    client = genai.GenerativeModel("gemini-pro")

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "persona_state" not in st.session_state:
    st.session_state.persona_state = {"step": 0, "data": {}, "retry": 0}

persona = st.session_state.persona_state

# --- Document Upload Section ---
st.sidebar.header("Upload Your Document")

uploaded_file = st.sidebar.file_uploader("Upload a PDF Document (e.g., bank statement, invoice)", type="pdf")

# Store the document as context once uploaded
if uploaded_file:
    # Display uploaded document name
    st.sidebar.write(f"Document uploaded: {uploaded_file.name}")
    # For demonstration purposes, display a placeholder image (could be a scan of the document)
    st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)
    # You can store the document in a variable and use it for AI queries later

# --- AI Query Section ---
st.sidebar.header("Ask the AI about your Document")

# AI Query Input
ai_query = st.text_input("What would you like to ask about this document?")

if ai_query:
    # Add the uploaded document as context for AI
    context = f"Context: The following document contains financial transactions. It includes bank statements or invoices, with relevant transaction details such as amounts, dates, descriptions, and references.\n\n"
    context += f"Document: {uploaded_file.name}\n"

    # Combine the context with the AI query
    query_prompt = f"{context}\nUser's query: {ai_query}"

    # Send the query to the selected AI provider
    if provider == "OpenAI":
        # OpenAI API to process the query with document context
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps with financial data analysis from documents."},
                {"role": "user", "content": query_prompt}
            ]
        ).choices[0].message.content
    else:  # For Google Gemini (if needed)
        response = client.generate_content(query_prompt).text

    # Display the AI response
    st.subheader("AI Response:")
    st.write(response)

# --- Main Application ---
# Sample workflow and question for context, to visualize document processing
st.markdown("""
    ## How It Works:
    1. Upload a document (such as a bank statement or invoice).
    2. The AI will analyze the document and allow you to ask questions about the extracted data.
    3. Receive AI-generated insights based on the content of the document.

    **Try asking about specific transactions, anomalies, or ask for a summary of the document's key details!**
""")

# Placeholder to show the document extraction process
if uploaded_file:
    # Simulate extracting data from the document and showing it as a table (in the future, this would come from OCR)
    st.subheader("Extracted Data (Simulated):")
    st.table({
        "Date": ["2025-07-01", "2025-07-02", "2025-07-03"],
        "Amount (£)": [6000, 12000, 4500],
        "Description": ["Transaction 1", "Transaction 2", "Transaction 3"],
        "Reference": ["ABC123", "DEF456", "GHI789"]
    })

    st.write("You can now ask the AI questions related to the data shown above!")

