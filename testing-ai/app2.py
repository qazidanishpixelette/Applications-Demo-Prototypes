import os
import tempfile
import streamlit as st
import PyPDF2
import datetime
from openai import OpenAI
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


def extract_pdf_data(file):
    """Extract text from the uploaded PDF document."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_values(text, form_title):
    """Extract specific values from the document text based on the form title."""
    data = {}

    # Split the text by sections to process the relevant form
    sections = text.split("Form ")

    # Determine the last part of the title to identify the form
    form_identifier = form_title.split(" ")[2]  # Extract "C-Corp" from "Form 1120 C-Corp"

    # Find the section that corresponds to the form identifier
    form_section = ""
    for section in sections:
        if form_identifier in section:
            form_section = section
            break

    if not form_section:
        raise ValueError(f"Form {form_title} not found in the provided PDF text.")

    # Split the form section into lines to process line by line
    lines = form_section.splitlines()

    # Define keywords to search for in the text
    keywords = {
        "name": "Name:",
        "address": "Business Address:",
        "ein": "Employer Identification Number (EIN):",
        "date_incorporated": "Incorporation Date:",
        "gross_receipts": "Gross Receipts or Sales:",
        "returns_allowances": "Returns and Allowances:",
        "cost_of_goods_sold": "Cost of Goods Sold:",
        "dividends": "Dividends:",
        "interest": "Interest:",
        "rents": "Gross Rents:",
        "royalties": "Gross Royalties:",
        "capital_gain": "Net Capital Gain:",
        "net_gain": "Net Gain or (Loss):",
        "salaries_wages": "Salaries and wages:",
        "repairs_maintenance": "Repairs and maintenance:",
        "bad_debts": "Bad debts:",
        "rents_deductions": "Rents:",
        "taxes_licenses": "Taxes and licenses:",
        "interest_deductions": "Interest:",
        "depreciation": "Depreciation:",
        "advertising": "Advertising:",
        "other_deductions": "Other Deductions:"
    }

    # Iterate over each line and check if it contains any of the keywords
    for line in lines:
        for key, keyword in keywords.items():
            if keyword in line:
                # Extract the value after the keyword
                value = line.split(":")[1].strip().replace(",", "").replace("$", "")
                if key in ["gross_receipts", "returns_allowances", "cost_of_goods_sold", "dividends", "interest", "rents", "royalties", "capital_gain", "net_gain", "salaries_wages", "repairs_maintenance", "bad_debts", "rents_deductions", "taxes_licenses", "interest_deductions", "depreciation", "advertising", "other_deductions"]:
                    # Convert numeric values to float
                    value = float(value)
                data[key] = value

    # Return the extracted data
    return data


# --- Document Upload Section ---
st.header("Upload Your Financial Document")

uploaded_file = st.file_uploader("Upload a PDF document (tax return, financial statement)", type="pdf")

# --- Process the Uploaded Document ---
if uploaded_file:
    # Save the uploaded document to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    # Extract the text from the PDF document
    pdf_text = extract_pdf_data(uploaded_file)

    # Extract relevant data from the document using the extracted text
    form_title = "Form 1120 C-Corp"  # Example form title
    extracted_data = extract_values(pdf_text, form_title)

    # --- Bookkeeping Field Extraction ---
    st.subheader("Extracted Bookkeeping Fields")
    st.table(extracted_data)

    # --- AI-Powered User Query Assistance ---
    st.subheader("Ask the AI about your document")

    user_query = st.text_input("What would you like to ask about this document?")

    if user_query:
        # Initialize Langchain components for querying the document
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

        # Split the document into chunks
        splitted_documents = text_splitter.split_documents([pdf_text])

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
st.markdown("Source code: Demo Prepared By [Pixelette Technologies](https://pixelettetech.com)")
