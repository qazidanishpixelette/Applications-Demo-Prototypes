import streamlit as st

# Title of the App
st.title("AI + OCR-Powered Financial Data Automation Assistant")

# Description
st.markdown("""
    This is a prototype demo for an AI-powered assistant that automates the extraction and analysis of financial data from documents like bank statements and invoices. You can upload a document, extract transactions, flag anomalies, and get AI-generated insights.
""")

# Sidebar for navigation options
st.sidebar.title("Options")

# File uploader for bank statement or invoice
uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type="pdf")

# Threshold for transaction extraction (e.g., £5,000)
threshold = st.sidebar.number_input("Transaction Threshold (£)", min_value=0, max_value=100000, value=5000, step=500)

# Button to trigger data extraction
if st.sidebar.button("Start Data Extraction"):
    if uploaded_file is not None:
        st.subheader("Document Uploaded: " + uploaded_file.name)
        # Show the uploaded document (This part would ideally be functional after OCR implementation)
        st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)
        
        # Placeholder for extracted data
        st.subheader("Extracted Data")
        st.markdown("**Transaction Details Table**")
        
        # Placeholder for the table with extracted data
        st.table({
            "Date": ["2025-07-01", "2025-07-02", "2025-07-03"],
            "Amount (£)": [6000, 12000, 4500],
            "Description": ["Transaction 1", "Transaction 2", "Transaction 3"],
            "Reference": ["ABC123", "DEF456", "GHI789"]
        })
        
        # Placeholder for AI-generated insights
        st.subheader("AI-Generated Insights")
        st.markdown("""
            - **Flagged Anomalies**: No significant anomalies detected for transactions over the threshold.
            - **General Insights**: Transactions seem to align with typical spending patterns. Consider reviewing large transactions for compliance.
        """)
        
        # Buttons to export data
        st.download_button(
            label="Export to CSV",
            data="sample_csv_data",  # Placeholder for actual CSV data
            file_name="extracted_data.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="Export to Tax Return Format",
            data="sample_tax_data",  # Placeholder for actual tax data export
            file_name="tax_return_format.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please upload a document to proceed.")
