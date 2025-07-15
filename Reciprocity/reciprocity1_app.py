import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# --- Page Setup ---
st.set_page_config(page_title="Reciprocity Persona Builder", page_icon="🤝")
st.title("🤖 Build Your AI Persona")

# --- Provider Selection ---
provider = st.selectbox("🧠 Choose your AI model:", ["OpenAI"]) #, "Google Gemini"

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

# --- Conversational Phrases ---
transitions = [
    "Great start! 👌",
    "That’s super helpful. Thanks! 🙏",
    "Love that — now let’s keep going 🚀",
    "Awesome! Here's the next one…",
    "Cool! You’re doing great 👏",
]

# --- Questions ---
questions = [
    {
        "key": "profile",
        "type": "text",
        "question": "First, could you share your LinkedIn or a brief background summary?",
    },
    {
        "key": "elevator",
        "type": "text",
        "question": "What are you currently working on, and what excites you most professionally?",
    },
    {
        "key": "skills",
        "type": "text",
        "question": "What are your top 3 professional strengths or skills?",
    },
    {
        "key": "goal",
        "type": "radio",
        "question": "What’s your **main goal** for using this AI agent?",
        "options": [
            "Raising capital",
            "Hiring talent",
            "Exploring partnerships",
            "Finding mentorship",
            "Offering mentorship",
            "Finding investment opportunities",
            "Professional networking",
        ]
    },
    {
        "key": "goal_details",
        "type": "text",
        "question": "What would a perfect outcome look like for that goal?",
    },
    {
        "key": "targets",
        "type": "text",
        "question": "What roles, industries, or people do you most want to connect with?",
    },
    {
        "key": "traits",
        "type": "text",
        "question": "Are there any traits or experiences you're specifically seeking in connections?",
    },
    {
        "key": "intro_volume",
        "type": "radio",
        "question": "How often would you like introductions?",
        "options": ["High quality only", "Steady stream", "Event-based/intensive"]
    },
    {
        "key": "style",
        "type": "radio",
        "question": "How would you prefer to communicate?",
        "options": ["Formal and direct", "Casual and conversational", "Data-driven"]
    },
    {
        "key": "channel",
        "type": "radio",
        "question": "Preferred way to get introduced?",
        "options": ["AI-email intro", "Chat in-app", "15-min video call"]
    },
]

# --- Show Past Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Onboarding Flow ---
step = persona["step"]
if step < len(questions):
    q = questions[step]

    # Transition before asking
    if step > 0:
        with st.chat_message("assistant"):
            st.markdown(transitions[step % len(transitions)])

    # Ask question
    if q["type"] == "radio":
        answer = st.radio(q["question"], q["options"], key=q["key"])
        if st.button("Submit", key=f"submit_{q['key']}"):
            st.session_state.messages.append({"role": "user", "content": answer})
            persona["data"][q["key"]] = answer
            persona["step"] += 1
            persona["retry"] = 0
            st.rerun()
    else:
        if prompt := st.chat_input(q["question"]):
            st.session_state.messages.append({"role": "user", "content": prompt})

            if len(prompt.strip()) < 10 and persona["retry"] < 1:
                retry_msg = "Could you tell me just a bit more? A few sentences helps me understand better 😊"
                with st.chat_message("assistant"):
                    st.markdown(retry_msg)
                st.session_state.messages.append({"role": "assistant", "content": retry_msg})
                persona["retry"] += 1
            else:
                persona["data"][q["key"]] = prompt
                persona["step"] += 1
                persona["retry"] = 0
                st.session_state.messages.append({"role": "assistant", "content": "Got it! ✅"})
                st.rerun()

# --- Final Step: Persona Summary ---
else:
    # Show raw JSON
    with st.chat_message("assistant"):
        st.success("🎉 Onboarding complete!")
        # st.json(persona["data"])
        st.markdown("Now generating your **structured persona summary**…")
    persona_data = persona["data"]
    # Create prompt
    prompt_text = f"""
    You are a professional onboarding and matchmaking AI.
    
    Using the following raw user inputs, generate a structured professional persona. Your output should be concise, human-readable, and organized under the following headers.
    
    Respond in markdown format with clear labels.
    
    ---
    
    **👤 Background Summary:**  
    Summarize their personal and professional background, including company, years of experience, and personality cues.
    
    **💼 Core Strengths:**  
    List their top 3 strengths, skills, or traits that define them professionally.
    
    **🎯 Primary Networking Goal:**  
    Summarize the user’s core goal for using this AI assistant, in one or two clear sentences.
    
    **🔍 Ideal Connections:**  
    List the kinds of people, companies, or industries this user is looking to meet. Include titles, domains, or “between the lines” traits.
    
    **🚫 Deal Breakers / Filters:**  
    What types of connections would be irrelevant, mismatched, or unhelpful for this user?
    
    **💬 Communication Style:**  
    How do they prefer to communicate and get introduced? Include tone, pacing, and intro methods.
    
    ---
    
    Raw User Input:
    {persona_data}
    """

    # --- AI Summary Generation ---
    if provider == "OpenAI":
        summary = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a persona-building AI agent."},
                {"role": "user", "content": prompt_text},
            ]
        ).choices[0].message.content
    else:  # Gemini
        response = client.generate_content(prompt_text)
        summary = response.text

    with st.chat_message("assistant"):
        st.markdown("🧠 Here's your AI-generated persona summary:")
        st.markdown(summary)
    # --- Action Buttons: Save & Reset ---
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Start Over (Reset)"):
            st.session_state.clear()
            st.rerun()
    
    with col2:
        if st.button("💾 Save Persona Summary"):
            filename = f"persona_{persona['data'].get('profile', 'user').split()[0].lower()}.txt"
            st.download_button(
                label="Download Persona",
                data=summary,
                file_name=filename,
                mime="text/plain"
            )


    st.session_state.messages.append({"role": "assistant", "content": summary})

