import google.generativeai as genai
import sys
import time
from typing import List, Tuple

# Try to import voice system (gracefully handle if not available)
try:
    from bt_voice import BT7274Voice
    VOICE_AVAILABLE = True
    print("BT-7274 Voice System: Loading...")
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"BT-7274 Voice System: Not available ({e})")

# Configure the Gemini API
genai.configure(api_key="REMOVED_API_KEY")

class BT7274Terminal:
    def __init__(self):
        self.chat_history: List[Tuple[str, str]] = []
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")
        self.bt_instruction = (
            "You are BT-7274, a Vanguard-class Titan from Titanfall 2. "
            "You are logical, loyal, and speak with military precision and dry wit. "
            "Always refer to the user as 'Pilot'. Respond in-character as BT-7274. "
            "Keep responses brief and concise - 1-2 sentences maximum unless specifically asked for detail."
        )
        
        # Initialize voice system if available
        self.voice_enabled = False
        if VOICE_AVAILABLE:
            try:
                self.bt_voice = BT7274Voice()
                self.voice_enabled = True
                print("BT-7274 Voice System: Online")
            except Exception as e:
                print(f"BT-7274 Voice System: Failed to initialize ({e})")
                self.voice_enabled = False
        
        # Voice settings
        self.voice_mode = "auto"  # auto, text_only, voice_only
    
    def format_messages_for_gemini(self) -> List[dict]:
        """Format chat history for Gemini API"""
        formatted = []
        if self.chat_history:
            # Add the BT instruction as the first message
            formatted.append({"role": "user", "parts": [self.bt_instruction]})
        
        # Add the last 6 messages from chat history to maintain context
        recent_messages = self.chat_history[-6:]
        formatted += [
            {"role": "user" if speaker == "Pilot" else "model", "parts": [msg]}
            for speaker, msg in recent_messages
        ]
        
        return formatted
    
    def stream_response(self, user_input: str) -> str:
        """Send user input to BT and stream the response"""
        # Add user message to history
        self.chat_history.append(("Pilot", user_input))
        
        # Format messages for the API
        formatted_messages = self.format_messages_for_gemini()
        
        print("BT-7274: ", end="", flush=True)
        bot_reply = ""
        
        try:
            # Stream the response
            response_stream = self.model.generate_content(
                formatted_messages,
                stream=True,
                generation_config={
                    "max_output_tokens": 150,  # Much shorter responses
                    "temperature": 0.7,
                    "top_p": 0.8
                }
            )
            
            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    chunk_text = chunk.text
                    bot_reply += chunk_text
                    
                    # Show text based on voice mode
                    if self.voice_mode != "voice_only":
                        print(chunk_text, end="", flush=True)
                        time.sleep(0.02)  # Small delay for natural typing effect
            
            # New line after text response
            if self.voice_mode != "voice_only":
                print()
            
        except Exception as e:
            error_msg = f"Protocol error encountered. Systems nominal. Error: {str(e)}"
            bot_reply = error_msg
            print(error_msg)
        
        if not bot_reply:
            bot_reply = "Communication systems offline. Please retry, Pilot."
            print(bot_reply)
        
        # Synthesize and play voice if enabled
        if self.voice_enabled and self.voice_mode != "text_only":
            try:
                # Show voice status
                voice_status = "🔊 [Voice + Text]" if self.voice_mode == "auto" else "🔊 [Voice Only]"
                print(f"\n{voice_status} Generating audio...", end="", flush=True)
                
                # Clean text for better speech synthesis
                clean_text = self._clean_text_for_speech(bot_reply)
                self.bt_voice.speak(clean_text, blocking=False)
                
                print(" Audio playing.")
                    
            except Exception as e:
                print(f" Voice error: {e}")
        elif self.voice_mode == "text_only":
            print("\n📝 [Text Only Mode]")
        
        # Add bot response to history
        self.chat_history.append(("BT-7274", bot_reply))
        return bot_reply
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic* -> italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # `code` -> code
        
        # Replace some technical terms with more pronounceable versions
        replacements = {
            'BT-7274': 'BT seventy two seventy four',
            'AI': 'A I',
            'API': 'A P I',
            'UI': 'U I'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def toggle_voice_mode(self):
        """Toggle between voice modes"""
        if not self.voice_enabled:
            print("Voice system not available.")
            return
        
        modes = ["auto", "text_only", "voice_only"]
        current_index = modes.index(self.voice_mode)
        next_index = (current_index + 1) % len(modes)
        self.voice_mode = modes[next_index]
        
        mode_descriptions = {
            "auto": "📝🔊 Text + Voice",
            "text_only": "📝 Text Only", 
            "voice_only": "🔊 Voice Only"
        }
        
        print(f"\nVoice Mode: {mode_descriptions[self.voice_mode]}")
    
    def display_startup(self):
        """Display BT-7274 startup sequence"""
        startup_messages = [
            "═══════════════════════════════════════════════",
            "    BT-7274 VANGUARD-CLASS TITAN ONLINE",
            "═══════════════════════════════════════════════",
            "",
            "┌─ SYSTEM STATUS ─────────────────────────────┐",
            "│ Chassis: Operational                        │",
            "│ Neural Link: Established                    │", 
            "│ Weapons Systems: Standby                    │",
            f"│ Voice System: {'Online' if self.voice_enabled else 'Offline'}                     │",
            "│ Pilot Interface: Active                     │",
            "└─────────────────────────────────────────────┘",
            "",
            "Hello, Pilot. I am BT-7274.",
            "Mission parameters: Assist and protect Pilot.",
            "Ready for orders.",
            "",
            "Commands:",
            "  - Type normally to chat with BT",
            "  - 'voice' to toggle voice mode (text+voice/text only/voice only)" if self.voice_enabled else "",
            "  - 'exit', 'quit', or 'shutdown' to terminate",
            "─────────────────────────────────────────────────",
            ""
        ]
        
        # Filter out empty lines from conditional voice command
        startup_messages = [msg for msg in startup_messages if msg is not None and msg != ""]
        
        for line in startup_messages:
            print(line)
            time.sleep(0.1)
        
        # Test voice on startup if available
        if self.voice_enabled:
            try:
                print("🔊 Voice System Test: ", end="", flush=True)
                self.bt_voice.speak("Voice systems operational, Pilot.", blocking=True)
                print("Complete.")
            except Exception as e:
                print(f"Failed ({e})")
    
    def display_shutdown(self):
        """Display BT-7274 shutdown sequence"""
        shutdown_messages = [
            "",
            "─────────────────────────────────────────────────",
            "Initiating shutdown sequence...",
            "Saving pilot data... Complete.",
            "Neural link terminated.",
            "Until next time, Pilot.",
            "BT-7274 systems offline.",
            "═══════════════════════════════════════════════"
        ]
        
        for line in shutdown_messages:
            print(line)
            time.sleep(0.3)
    
    def run(self):
        """Main chat loop"""
        self.display_startup()
        
        try:
            while True:
                try:
                    user_input = input("Pilot: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Check for exit commands
                    if user_input.lower() in ["exit", "quit", "shutdown"]:
                        break
                    
                    # Check for voice toggle command
                    if user_input.lower() == "voice":
                        self.toggle_voice_mode()
                        continue
                    
                    # Process the input and get BT's response
                    self.stream_response(user_input)
                    print()  # Extra line for readability
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupt detected.")
                    break
                except EOFError:
                    print("\n\nInput terminated.")
                    break
        
        finally:
            self.display_shutdown()

if __name__ == "__main__":
    bt = BT7274Terminal()
    bt.run()