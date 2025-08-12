import streamlit as st
import PyPDF2
import re
import datetime
import tempfile
import os
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI as LangChainOpenAI

# --- Page Setup ---
st.set_page_config(page_title="AI-Powered PDF Tax Form Assistant", page_icon="🤖", layout="wide")

# --- API Keys ---
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")

# --- Client Init ---
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = LangChainOpenAI(temperature=0, openai_api_key=openai_api_key)
else:
    st.error("OpenAI API Key is missing. Please add it to your Streamlit secrets.")
    st.stop()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {}
if "insights" not in st.session_state:
    st.session_state.insights = ""

st.title("🤖 AI-Powered PDF Tax Form Assistant")
st.markdown("Upload a PDF tax return document to extract fields, ask questions, and get AI-powered insights.")

def extract_pdf_text(pdf_file) -> str:
    """Extract all text from the uploaded PDF document."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def create_vector_store(pdf_file):
    """Create vector store from uploaded PDF for AI querying."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and split document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(splits, embeddings)
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return vector_store, qa_chain
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None, None

def extract_form_fields(text: str, form_type: str = "1120") -> Dict[str, Any]:
    """Extract key fields from the PDF text based on common tax form patterns."""
    fields = {}
    
    # Basic Information Fields
    basic_patterns = {
        "name": [r"Name.*?:\s*([^\n]+)", r"Corporation Name.*?:\s*([^\n]+)", r"Business Name.*?:\s*([^\n]+)"],
        "address": [r"Address.*?:\s*([^\n]+)", r"Business Address.*?:\s*([^\n]+)", r"Street Address.*?:\s*([^\n]+)"],
        "ein": [r"EIN.*?:\s*(\d{2}-\d{7})", r"Employer.*?Number.*?:\s*(\d{2}-\d{7})", r"(\d{2}-\d{7})"],
        "city_state_zip": [r"City.*?State.*?ZIP.*?:\s*([^\n]+)", r"City.*?:\s*([^\n]+)"],
        "incorporation_date": [r"Incorporation.*?Date.*?:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", r"Date.*?Incorporated.*?:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"]
    }
    
    # Income Fields
    income_patterns = {
        "gross_receipts": [r"Gross.*?[Rr]eceipts.*?:\s*\$?([\d,]+\.?\d*)", r"Total.*?[Ii]ncome.*?:\s*\$?([\d,]+\.?\d*)"],
        "dividends": [r"Dividends.*?:\s*\$?([\d,]+\.?\d*)", r"Dividend.*?[Ii]ncome.*?:\s*\$?([\d,]+\.?\d*)"],
        "interest": [r"Interest.*?:\s*\$?([\d,]+\.?\d*)", r"Interest.*?[Ii]ncome.*?:\s*\$?([\d,]+\.?\d*)"],
        "capital_gains": [r"Capital.*?[Gg]ains.*?:\s*\$?([\d,]+\.?\d*)", r"Net.*?[Gg]ain.*?:\s*\$?([\d,]+\.?\d*)"],
        "other_income": [r"Other.*?[Ii]ncome.*?:\s*\$?([\d,]+\.?\d*)", r"Miscellaneous.*?[Ii]ncome.*?:\s*\$?([\d,]+\.?\d*)"]
    }
    
    # Deduction Fields
    deduction_patterns = {
        "cost_of_goods_sold": [r"Cost.*?[Gg]oods.*?[Ss]old.*?:\s*\$?([\d,]+\.?\d*)", r"COGS.*?:\s*\$?([\d,]+\.?\d*)"],
        "salaries_wages": [r"Salaries.*?[Ww]ages.*?:\s*\$?([\d,]+\.?\d*)", r"Compensation.*?:\s*\$?([\d,]+\.?\d*)"],
        "rent": [r"Rent.*?:\s*\$?([\d,]+\.?\d*)", r"Rental.*?[Ee]xpense.*?:\s*\$?([\d,]+\.?\d*)"],
        "depreciation": [r"Depreciation.*?:\s*\$?([\d,]+\.?\d*)", r"Amortization.*?:\s*\$?([\d,]+\.?\d*)"],
        "advertising": [r"Advertising.*?:\s*\$?([\d,]+\.?\d*)", r"Marketing.*?:\s*\$?([\d,]+\.?\d*)"],
        "office_expenses": [r"Office.*?[Ee]xpenses.*?:\s*\$?([\d,]+\.?\d*)", r"Administrative.*?:\s*\$?([\d,]+\.?\d*)"],
        "professional_fees": [r"Professional.*?[Ff]ees.*?:\s*\$?([\d,]+\.?\d*)", r"Legal.*?[Ff]ees.*?:\s*\$?([\d,]+\.?\d*)"],
        "insurance": [r"Insurance.*?:\s*\$?([\d,]+\.?\d*)", r"Insurance.*?[Pp]remiums.*?:\s*\$?([\d,]+\.?\d*)"],
        "utilities": [r"Utilities.*?:\s*\$?([\d,]+\.?\d*)", r"Electric.*?Gas.*?:\s*\$?([\d,]+\.?\d*)"],
        "other_deductions": [r"Other.*?[Dd]eductions.*?:\s*\$?([\d,]+\.?\d*)", r"Miscellaneous.*?[Dd]eductions.*?:\s*\$?([\d,]+\.?\d*)"]
    }
    
    # Tax Computation Fields
    tax_patterns = {
        "taxable_income": [r"Taxable.*?[Ii]ncome.*?:\s*\$?([\d,]+\.?\d*)", r"Income.*?[Ss]ubject.*?[Tt]ax.*?:\s*\$?([\d,]+\.?\d*)"],
        "income_tax": [r"Income.*?[Tt]ax.*?:\s*\$?([\d,]+\.?\d*)", r"Federal.*?[Tt]ax.*?:\s*\$?([\d,]+\.?\d*)"],
        "estimated_payments": [r"Estimated.*?[Pp]ayments.*?:\s*\$?([\d,]+\.?\d*)", r"Quarterly.*?[Pp]ayments.*?:\s*\$?([\d,]+\.?\d*)"],
        "amount_owed": [r"Amount.*?[Oo]wed.*?:\s*\$?([\d,]+\.?\d*)", r"Balance.*?[Dd]ue.*?:\s*\$?([\d,]+\.?\d*)"],
        "refund": [r"Refund.*?:\s*\$?([\d,]+\.?\d*)", r"Overpayment.*?:\s*\$?([\d,]+\.?\d*)"]
    }
    
    # Combine all patterns
    all_patterns = {**basic_patterns, **income_patterns, **deduction_patterns, **tax_patterns}
    
    # Extract fields using regex patterns
    for field_name, patterns in all_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                # Clean up monetary values
                if field_name not in ["name", "address", "ein", "city_state_zip", "incorporation_date"]:
                    value = value.replace(",", "").replace("$", "")
                    try:
                        value = float(value)
                    except:
                        value = 0.0
                fields[field_name] = value
                break
        
        # Set default values if not found
        if field_name not in fields:
            if field_name in ["name", "address", "ein", "city_state_zip"]:
                fields[field_name] = ""
            elif field_name == "incorporation_date":
                fields[field_name] = None
            else:
                fields[field_name] = 0.0
    
    return fields

def generate_insights(extracted_data: Dict[str, Any]) -> str:
    """Generate AI-powered insights from extracted tax data."""
    try:
        # Prepare data summary for analysis
        total_income = sum([
            extracted_data.get("gross_receipts", 0),
            extracted_data.get("dividends", 0),
            extracted_data.get("interest", 0),
            extracted_data.get("capital_gains", 0),
            extracted_data.get("other_income", 0)
        ])
        
        total_deductions = sum([
            extracted_data.get("cost_of_goods_sold", 0),
            extracted_data.get("salaries_wages", 0),
            extracted_data.get("rent", 0),
            extracted_data.get("depreciation", 0),
            extracted_data.get("advertising", 0),
            extracted_data.get("office_expenses", 0),
            extracted_data.get("professional_fees", 0),
            extracted_data.get("insurance", 0),
            extracted_data.get("utilities", 0),
            extracted_data.get("other_deductions", 0)
        ])
        
        # Create prompt for insights generation
        prompt = f"""
        Analyze the following tax return data and provide concise financial insights:
        
        Company: {extracted_data.get('name', 'N/A')}
        EIN: {extracted_data.get('ein', 'N/A')}
        
        INCOME BREAKDOWN:
        - Gross Receipts: ${extracted_data.get('gross_receipts', 0):,.2f}
        - Dividends: ${extracted_data.get('dividends', 0):,.2f}
        - Interest: ${extracted_data.get('interest', 0):,.2f}
        - Capital Gains: ${extracted_data.get('capital_gains', 0):,.2f}
        - Other Income: ${extracted_data.get('other_income', 0):,.2f}
        - Total Income: ${total_income:,.2f}
        
        DEDUCTIONS BREAKDOWN:
        - Cost of Goods Sold: ${extracted_data.get('cost_of_goods_sold', 0):,.2f}
        - Salaries & Wages: ${extracted_data.get('salaries_wages', 0):,.2f}
        - Rent: ${extracted_data.get('rent', 0):,.2f}
        - Depreciation: ${extracted_data.get('depreciation', 0):,.2f}
        - Advertising: ${extracted_data.get('advertising', 0):,.2f}
        - Professional Fees: ${extracted_data.get('professional_fees', 0):,.2f}
        - Insurance: ${extracted_data.get('insurance', 0):,.2f}
        - Total Deductions: ${total_deductions:,.2f}
        
        TAX COMPUTATION:
        - Taxable Income: ${extracted_data.get('taxable_income', 0):,.2f}
        - Income Tax: ${extracted_data.get('income_tax', 0):,.2f}
        
        Please provide:
        1. Key financial highlights (3-4 bullet points)
        2. Notable observations or potential areas of concern
        3. Expense analysis (largest expense categories)
        4. Tax efficiency observations
        
        Keep the analysis concise and actionable.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing insights on tax return data. Be concise and focus on key business metrics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def ask_question(question: str, qa_chain, chat_history: List[Tuple[str, str]]) -> str:
    """Process user questions about the uploaded document."""
    try:
        if qa_chain is None:
            return "Please upload a document first to ask questions about it."
        
        # Get response from QA chain
        result = qa_chain({
            "question": question,
            "chat_history": chat_history
        })
        
        return result["answer"]
    except Exception as e:
        return f"Error processing question: {str(e)}"

def display_form(extracted_fields: Dict[str, Any]):
    """Display a Streamlit form with extracted fields for user review and editing."""
    
    with st.form("tax_form"):
        st.header("📊 Extracted Tax Form Data")
        st.markdown("Review and edit the extracted information below:")
        
        # Basic Information Section
        st.subheader("🏢 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Corporation Name", value=extracted_fields.get("name", ""))
            address = st.text_input("Business Address", value=extracted_fields.get("address", ""))
            city_state_zip = st.text_input("City, State, ZIP", value=extracted_fields.get("city_state_zip", ""))
        
        with col2:
            ein = st.text_input("Employer Identification Number (EIN)", value=extracted_fields.get("ein", ""))
            incorporation_date = st.date_input("Incorporation Date", 
                                             value=extracted_fields.get("incorporation_date", datetime.date.today()))
        
        # Income Section
        st.subheader("💰 Income")
        col3, col4 = st.columns(2)
        
        with col3:
            gross_receipts = st.number_input("Gross Receipts", value=float(extracted_fields.get("gross_receipts", 0)), format="%.2f")
            dividends = st.number_input("Dividends", value=float(extracted_fields.get("dividends", 0)), format="%.2f")
            interest = st.number_input("Interest Income", value=float(extracted_fields.get("interest", 0)), format="%.2f")
        
        with col4:
            capital_gains = st.number_input("Capital Gains", value=float(extracted_fields.get("capital_gains", 0)), format="%.2f")
            other_income = st.number_input("Other Income", value=float(extracted_fields.get("other_income", 0)), format="%.2f")
        
        # Deductions Section
        st.subheader("📉 Deductions")
        col5, col6 = st.columns(2)
        
        with col5:
            cost_of_goods_sold = st.number_input("Cost of Goods Sold", value=float(extracted_fields.get("cost_of_goods_sold", 0)), format="%.2f")
            salaries_wages = st.number_input("Salaries and Wages", value=float(extracted_fields.get("salaries_wages", 0)), format="%.2f")
            rent = st.number_input("Rent", value=float(extracted_fields.get("rent", 0)), format="%.2f")
            depreciation = st.number_input("Depreciation", value=float(extracted_fields.get("depreciation", 0)), format="%.2f")
            advertising = st.number_input("Advertising", value=float(extracted_fields.get("advertising", 0)), format="%.2f")
        
        with col6:
            office_expenses = st.number_input("Office Expenses", value=float(extracted_fields.get("office_expenses", 0)), format="%.2f")
            professional_fees = st.number_input("Professional Fees", value=float(extracted_fields.get("professional_fees", 0)), format="%.2f")
            insurance = st.number_input("Insurance", value=float(extracted_fields.get("insurance", 0)), format="%.2f")
            utilities = st.number_input("Utilities", value=float(extracted_fields.get("utilities", 0)), format="%.2f")
            other_deductions = st.number_input("Other Deductions", value=float(extracted_fields.get("other_deductions", 0)), format="%.2f")
        
        # Tax Computation Section
        st.subheader("🧮 Tax Computation")
        col7, col8 = st.columns(2)
        
        with col7:
            taxable_income = st.number_input("Taxable Income", value=float(extracted_fields.get("taxable_income", 0)), format="%.2f")
            income_tax = st.number_input("Income Tax", value=float(extracted_fields.get("income_tax", 0)), format="%.2f")
            estimated_payments = st.number_input("Estimated Tax Payments", value=float(extracted_fields.get("estimated_payments", 0)), format="%.2f")
        
        with col8:
            amount_owed = st.number_input("Amount Owed", value=float(extracted_fields.get("amount_owed", 0)), format="%.2f")
            refund = st.number_input("Refund", value=float(extracted_fields.get("refund", 0)), format="%.2f")
        
        # Other Information Section
        st.subheader("ℹ️ Other Information")
        filing_status = st.selectbox("Filing Status", ["Regular Corporation", "Personal Service Corporation", "Other"])
        accounting_method = st.radio("Accounting Method", ["Cash", "Accrual", "Other"])
        
        # Submit button
        submitted = st.form_submit_button("✅ Submit Form Data")
        
        if submitted:
            st.success("✅ Form data has been successfully processed!")
            st.balloons()
            
            # Display summary
            st.subheader("📋 Form Summary")
            st.write(f"**Corporation Name:** {name}")
            st.write(f"**EIN:** {ein}")
            st.write(f"**Gross Receipts:** ${gross_receipts:,.2f}")
            st.write(f"**Total Deductions:** ${(cost_of_goods_sold + salaries_wages + rent + depreciation + advertising + office_expenses + professional_fees + insurance + utilities + other_deductions):,.2f}")
            st.write(f"**Taxable Income:** ${taxable_income:,.2f}")
            st.write(f"**Income Tax:** ${income_tax:,.2f}")

# Main App Interface
uploaded_file = st.file_uploader("Choose a PDF tax return file", type="pdf")

if uploaded_file is not None:
    st.success("✅ PDF file uploaded successfully!")
    
    # Extract text from PDF
    with st.spinner("Processing PDF and setting up AI capabilities..."):
        pdf_text = extract_pdf_text(uploaded_file)
        
        # Create vector store for AI querying
        vector_store, qa_chain = create_vector_store(uploaded_file)
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
    
    if pdf_text:
        st.success("✅ PDF processing completed!")
        
        # Extract form fields
        with st.spinner("Extracting form fields and generating insights..."):
            extracted_fields = extract_form_fields(pdf_text)
            st.session_state.extracted_data = extracted_fields
            
            # Generate AI insights
            insights = generate_insights(extracted_fields)
            st.session_state.insights = insights
        
        st.success(f"✅ Analysis completed! Found {len([v for v in extracted_fields.values() if v])} fields with data.")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Extracted Data", "🤖 AI Assistant", "💡 Insights", "📄 Document Preview"])
        
        with tab1:
            # Display the form
            display_form(extracted_fields)
        
        with tab2:
            st.header("🤖 AI Document Assistant")
            st.markdown("Ask questions about your tax document. I can help you find specific information, analyze data, and clarify details.")
            
            # Example questions
            st.markdown("**Example questions:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💰 What is the total tax paid?"):
                    st.session_state.current_question = "What is the total tax paid according to this document?"
                if st.button("📊 List all expenses above $5000"):
                    st.session_state.current_question = "List all expenses or deductions above $5000 from this tax return"
            with col2:
                if st.button("🔍 Are there any anomalies?"):
                    st.session_state.current_question = "Are there any unusual patterns or potential anomalies in this tax return?"
                if st.button("💼 What's the largest expense category?"):
                    st.session_state.current_question = "What is the largest expense or deduction category in this tax return?"
            
            # Chat interface
            user_question = st.text_input("Ask a question about your document:", 
                                        value=st.session_state.get("current_question", ""))
            
            if user_question and st.button("Ask Question"):
                with st.spinner("Analyzing document..."):
                    answer = ask_question(user_question, st.session_state.qa_chain, st.session_state.chat_history)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, answer))
                    
                    # Clear current question
                    if "current_question" in st.session_state:
                        del st.session_state.current_question
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("💬 Conversation History")
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {question[:50]}..."):
                        st.write(f"**Question:** {question}")
                        st.write(f"**Answer:** {answer}")
                
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        with tab3:
            st.header("💡 AI-Generated Insights")
            st.markdown("AI analysis of your tax document highlighting key financial metrics and observations:")
            
            if st.session_state.insights:
                st.markdown(st.session_state.insights)
            else:
                st.info("No insights available. Please ensure the document was processed correctly.")
            
            if st.button("🔄 Regenerate Insights"):
                with st.spinner("Generating fresh insights..."):
                    new_insights = generate_insights(st.session_state.extracted_data)
                    st.session_state.insights = new_insights
                    st.rerun()
        
        with tab4:
            # Show extracted text preview
            st.header("📄 Document Preview")
            st.markdown("Preview of extracted text from your PDF:")
            st.text_area("PDF Content", pdf_text[:3000] + "..." if len(pdf_text) > 3000 else pdf_text, height=400)
    
    else:
        st.error("❌ Could not extract text from the PDF. Please check if the file is valid.")

else:
    st.info("👆 Please upload a PDF tax return document to get started.")
    
    # Show sample instructions
    st.markdown("""
    ### 🚀 Features:
    
    #### 📊 **Smart Field Extraction**
    - Automatically extracts key tax form fields
    - Supports various tax forms (1120, 1040, etc.)
    - Editable form interface for corrections
    
    #### 🤖 **AI-Powered Assistant**
    - Ask natural language questions about your document
    - Get contextual answers based on document content
    - Maintains conversation history for follow-up questions
    
    #### 💡 **Intelligent Insights**
    - AI-generated financial analysis and summaries
    - Anomaly detection and key observations
    - Expense category analysis and tax efficiency insights
    
    #### 📋 **Supported Fields:**
    - **Basic Info:** Name, Address, EIN, Incorporation Date
    - **Income:** Gross Receipts, Dividends, Interest, Capital Gains
    - **Deductions:** COGS, Salaries, Rent, Depreciation, Advertising, etc.
    - **Tax Computation:** Taxable Income, Tax Owed, Payments, Refunds
    
    ### 💰 **Budget-Optimized Design**
    This demo is designed for efficient AI usage with cost-effective API calls while maintaining high functionality.
    """)

