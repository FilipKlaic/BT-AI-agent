import streamlit as st
import google.generativeai as genai

genai.configure(api_key="REMOVED_API_KEY")  # <-- Add this line

st.set_page_config(page_title="Gemini AI Chatbot", page_icon="🤖", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #00bfff;'>🤖 Gemini AI Chatbot</h1>",
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message...")

# Only stream if user_input is present
if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Gemini is thinking..."):
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        bt_instruction = (
            "You are BT-7274, a Vanguard-class Titan from Titanfall 2. "
            "You are logical, loyal, and speak with military precision and dry wit. "
            "Always refer to the user as 'Pilot'. Respond in-character as BT-7274."
        )
        # Only prepend the instruction if there is at least one user message
        formatted = []
        if st.session_state.chat_history:
            formatted.append({"role": "user", "parts": [bt_instruction]})
        formatted += [
            {"role": "user" if speaker == "You" else "model", "parts": [msg]}
            for speaker, msg in st.session_state.chat_history[-6:]
        ]
        bot_reply = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                for chunk in model.generate_content(
                    formatted,
                    stream=True,
                    generation_config={"max_output_tokens": 512}
                ):
                    if hasattr(chunk, "text") and chunk.text:
                        bot_reply += chunk.text
                        message_placeholder.markdown(bot_reply)
            except StopIteration:
                message_placeholder.markdown("_[No response from model]_")
        st.session_state.chat_history.append(("Bot", bot_reply if bot_reply else "[No response]"))

# Show all previous messages except the last assistant message if just streamed
history = st.session_state.chat_history
if user_input:
    # Don't show the last assistant message (just streamed)
    display_history = history[:-1]
else:
    display_history = history

for speaker, message in display_history:
    if speaker == "You":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)