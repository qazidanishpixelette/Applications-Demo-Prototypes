import streamlit as st
from openai import OpenAI

# --- Configuration ---
st.set_page_config(page_title="Reciprocity Onboarding", page_icon="🤝")
st.title("🤖 Reciprocity Persona Onboarding")

st.markdown("""
Welcome! This AI assistant will help you build a **personalized professional persona** so it can later find the most relevant people, partners, or opportunities for you.

**Let’s get started!** 👇
""")

# --- API Key ---
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to begin.", icon="⚠️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# --- Session Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "persona_state" not in st.session_state:
    st.session_state.persona_state = {
        "step": 0,
        "data": {},
    }

persona = st.session_state.persona_state

# --- Onboarding Questions ---
onboarding_steps = [
    {"key": "profile", "question": "📎 Can you share a link to your LinkedIn profile or a short summary of your background?"},
    {"key": "elevator", "question": "🚀 What are you currently working on, and what are you most passionate about?"},
    {"key": "skills", "question": "💡 What are your top 3 skills or professional strengths?"},
    {"key": "goal", "question": "🎯 What is your **main goal** for using this AI agent? (e.g., raising capital, finding collaborators, recruiting, etc.)"},
    {"key": "goal_details", "question": "📐 Can you describe what a 'perfect outcome' would look like for that goal?"},
    {"key": "targets", "question": "👥 What kinds of people, roles, or industries are most relevant for you to connect with?"},
    {"key": "traits", "question": "🔍 Beyond job titles, what qualities or experiences are you looking for in your ideal match?"},
    {"key": "dealbreakers", "question": "⛔ Are there any absolute deal breakers or people you’d prefer not to be introduced to?"},
    {"key": "intro_volume", "question": "📬 How often would you like to be introduced to people? (Low-volume/high-quality, steady flow, event-based, etc.)"},
    {"key": "style", "question": "💬 How would you describe your preferred communication style? (Casual, direct, data-driven, etc.)"},
    {"key": "channel", "question": "🔗 How should introductions happen? (In-app chat, email, scheduled call?)"},
]

# --- Dialogue Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Flow ---
if prompt := st.chat_input("Type your response here..."):

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    current_step = persona["step"]
    if current_step < len(onboarding_steps):
        key = onboarding_steps[current_step]["key"]
        persona["data"][key] = prompt
        persona["step"] += 1

        if persona["step"] < len(onboarding_steps):
            next_q = onboarding_steps[persona["step"]]["question"]
            with st.chat_message("assistant"):
                st.markdown(next_q)
            st.session_state.messages.append({"role": "assistant", "content": next_q})
        else:
            # Onboarding complete
            with st.chat_message("assistant"):
                st.success("🎉 You're all set! Your personalized AI agent has been trained.")
                st.markdown("Here’s a preview of your **persona profile**:")
                st.json(persona["data"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": "🎉 You're all set! Your personalized AI agent has been trained.\n\n" +
                           "Here’s a preview of your **persona profile**:\n" +
                           str(persona["data"])
            })
    else:
        with st.chat_message("assistant"):
            st.info("You’ve already completed onboarding. Refresh the page to start over.")
