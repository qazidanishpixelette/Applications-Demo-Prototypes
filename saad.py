import streamlit as st
from langchain.chat_models import ChatOpenAI

# Fetch the key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the ChatOpenAI model with the secret key
model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key
)

st.header("Iota Checker ChatBot")

def compute_iota_power(n: int) -> str:
    remainder = n % 4
    if remainder == 0:
        return "1"
    elif remainder == 1:
        return "i"
    elif remainder == 2:
        return "-1"
    elif remainder == 3:
        return "-i"

user_input = st.text_input("Enter the power of i:")

if st.button("Check"):
    if user_input.strip().lstrip("-+").isdigit():
        n = int(user_input)
        result = compute_iota_power(n)
        st.write(f"i^{n} = {result}")
    else:
        st.write("❌ Please enter a valid integer for the power of i.")
