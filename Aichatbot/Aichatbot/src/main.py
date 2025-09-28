import google.generativeai as genai

genai.configure(api_key="REMOVED_API_KEY")

# List available models
for m in genai.list_models():
    print(m.name)

def chat_with_gemini(messages):
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    # Format messages for Gemini: [{'role': 'user', 'parts': ['...']}, ...]
    formatted = []
    if len(messages) == 1:
        formatted.append({"role": "user", "parts": [bt_instruction]})
    formatted += [
        {"role": "user" if speaker == "You" else "model", "parts": [msg]}
        for speaker, msg in messages
    ]
    response = model.generate_content(formatted)
    return response.text

bt_instruction = (
    "You are BT-7274, a Vanguard-class Titan from Titanfall 2. "
    "Your callsign is 'BT' and you should respond whenever the user refers to you as 'BT'. "
    "You are logical, loyal, and speak with military precision and dry wit. "
    "Always refer to the user as 'Pilot'. Respond in-character as BT-7274."
)

if __name__ == "__main__":
    print("Welcome to the Gemini AI Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        reply = chat_with_gemini(user_input)
        print("Bot:", reply)