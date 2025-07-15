import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# --- Setup ---
st.set_page_config(page_title="Reciprocity AI Onboarding", page_icon="🤝")
st.title("🤖 Build Your AI Persona")

# --- Select Provider ---
provider = st.selectbox("🧠 Choose your AI model:", ["OpenAI", "Gemini"])

openai_api_key = st.secrets.get("openai_api_key")
gemini_api_key = st.secrets.get("gemini_api")

# --- Model Setup ---
if provider == "OpenAI":
    if not openai_api_key:
        st.error("Missing OpenAI API key.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)
elif provider == "Gemini":
    if not gemini_api_key:
        st.error("Missing Gemini API key.")
        st.stop()
    genai.configure(api_key=gemini_api_key)
    client = genai.GenerativeModel("gemini-pro")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "persona_state" not in st.session_state:
    st.session_state.persona_state = {"step": 0, "data": {}, "retry": 0}

persona = st.session_state.persona_state

# --- Question Set ---
questions = [
    {
        "key": "profile",
        "type": "text",
        "question": "Let’s kick off with a quick intro — could you share your LinkedIn or a few lines about your background?",
    },
    {
        "key": "elevator",
        "type": "text",
        "question": "Awesome! What are you currently working on and most excited about professionally?",
    },
    {
        "key": "skills",
        "type": "text",
        "question": "Great! What would you say are your top 3 strengths or areas of expertise?",
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
        "question": "Tell me more about what a perfect outcome would look like for that goal.",
    },
    {
        "key": "targets",
        "type": "text",
        "question": "What kinds of people, industries, or companies are most relevant for you to connect with?",
    },
    {
        "key": "traits",
        "type": "text",
        "question": "Beyond job titles, are there any traits or backgrounds you're especially looking for in a match?",
    },
    {
        "key": "intro_volume",
        "type": "radio",
        "question": "How frequently would you like introductions?",
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
        "question": "Preferred introduction method?",
        "options": ["AI-email intro", "Chat in-app", "15-min video call"]
    },
]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Ask Current Question ---
step = persona["step"]
if step < len(questions):
    q = questions[step]

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

            # Heuristic for vague or short responses
            if len(prompt.strip()) < 10 and persona["retry"] < 1:
                retry_msg = "Hmm, could you tell me a bit more? A sentence or two would be helpful! 😊"
                with st.chat_message("assistant"):
                    st.markdown(retry_msg)
                st.session_state.messages.append({"role": "assistant", "content": retry_msg})
                persona["retry"] += 1
            else:
                # Save and move to next
                persona["data"][q["key"]] = prompt
                persona["step"] += 1
                persona["retry"] = 0
                with st.chat_message("assistant"):
                    st.markdown("Thanks! Got it ✅")
                st.session_state.messages.append({"role": "assistant", "content": "Thanks! Got it ✅"})
                st.rerun()
else:
    # --- Persona Summary & Updates ---
    with st.chat_message("assistant"):
        st.success("🎉 All done! Here's your personalized profile:")
        st.json(persona["data"])
        st.markdown("If you'd like to **update or add anything**, just type it below! I'll include it in your profile.")

    if update := st.chat_input("Want to add or clarify something?"):
        st.session_state.messages.append({"role": "user", "content": update})
        with st.chat_message("user"):
            st.markdown(update)

        # Very basic merge logic
        persona["data"]["additional_notes"] = persona["data"].get("additional_notes", "") + " " + update
        with st.chat_message("assistant"):
            st.markdown("Got it — I've added that to your profile 📝")
        st.session_state.messages.append({"role": "assistant", "content": "Got it — I've added that to your profile 📝"})
