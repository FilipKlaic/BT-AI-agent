from google import genai
from google.genai import types
import sys
import time
import os
import re
from typing import List, Tuple
from dotenv import load_dotenv

import config

# Load environment variables
load_dotenv()

# Try to import voice system (gracefully handle if not available)
try:
    # Try Piper TTS first (proper way to use the BT-7274 model)
    from piper_voice import PiperBTVoice, check_piper_setup
    if check_piper_setup():
        from piper_voice import PiperBTVoice as BT7274Voice
        VOICE_AVAILABLE = True
        VOICE_TYPE = "piper"
        print("BT-7274 Voice System: Loading with Piper TTS...")
    else:
        raise ImportError("Piper not available")
except ImportError:
    try:
        # Fall back to the ONNX + Windows TTS hybrid
        from bt_voice import BT7274Voice
        VOICE_AVAILABLE = True
        VOICE_TYPE = "hybrid"
        print("BT-7274 Voice System: Loading with fallback system...")
    except ImportError as e:
        VOICE_AVAILABLE = False
        VOICE_TYPE = None
        print(f"BT-7274 Voice System: Not available ({e})")

# Configure the Gemini API
api_key = config.GEMINI_API_KEY
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

class BT7274Terminal:
    def __init__(self):
        self.chat_history: List[Tuple[str, str]] = []
        self.bt_instruction = config.BT_INSTRUCTION
        
        # Initialize voice system if available
        self.voice_enabled = False
        self.voice_type = VOICE_TYPE
        if VOICE_AVAILABLE:
            try:
                self.bt_voice = BT7274Voice()
                self.voice_enabled = True
                voice_status = "Piper TTS" if VOICE_TYPE == "piper" else "Hybrid (ONNX+TTS)"
                print(f"BT-7274 Voice System: Online ({voice_status})")
            except Exception as e:
                print(f"BT-7274 Voice System: Failed to initialize ({e})")
                self.voice_enabled = False
        
        # Voice settings
        self.voice_mode = "auto"  # auto, text_only, voice_only
    
    def format_messages_for_gemini(self) -> List[dict]:
        """Format chat history for Gemini API"""
        contents = []

        # Add the last 6 messages from chat history to maintain context
        recent_messages = self.chat_history[-6:]
        for speaker, msg in recent_messages:
            contents.append(types.Content(
                role="user" if speaker == "Pilot" else "model",
                parts=[types.Part.from_text(text=msg)]
            ))
        
        return contents

    def stream_response(self, user_input: str) -> str:
        """Send user input to BT and stream the response"""
        # Add user message to history
        self.chat_history.append(("Pilot", user_input))
        
        # Format messages for the API
        contents = self.format_messages_for_gemini()
        
        print("BT-7274: ", end="", flush=True)
        bot_reply = ""
        
        try:
            # Stream the response
            response_stream = client.models.generate_content_stream(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.bt_instruction,
                    max_output_tokens=500,  # Increased token limit
                    temperature=0.7,
                    top_p=0.8
                )
            )
            
            for chunk in response_stream:
                if chunk.text:
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
                voice_status = "[Voice + Text]" if self.voice_mode == "auto" else "[Voice Only]"
                print(f"\n{voice_status} Generating audio...", end="", flush=True)
                
                # Clean text for better speech synthesis
                clean_text = self._clean_text_for_speech(bot_reply)
                self.bt_voice.speak(clean_text, blocking=False)
                
                print(" Audio playing.")
                    
            except Exception as e:
                print(f" Voice error: {e}")
        elif self.voice_mode == "text_only":
            print("\n[Text Only Mode]")
        
        # Add bot response to history
        self.chat_history.append(("BT-7274", bot_reply))
        return bot_reply
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic* -> italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # `code` -> code
        
        # Replace some technical terms using config
        for old, new in config.VOICE_REPLACEMENTS.items():
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
            "auto": "Text + Voice",
            "text_only": "Text Only", 
            "voice_only": "Voice Only"
        }
        
        print(f"\nVoice Mode: {mode_descriptions[self.voice_mode]}")
    
    def display_startup(self):
        """Display BT-7274 startup sequence"""
        # Use messages from config
        startup_messages = list(config.STARTUP_MESSAGES)
        
        # Insert dynamic status
        try:
             # Find where to insert status
             weapons_line = next((i for i, line in enumerate(startup_messages) if "Weapons Systems: Standby" in line), -1)
             if weapons_line != -1:
                 # Calculate padding dynamically or just use fixed width (45 chars inside)
                 status_text = "Online" if self.voice_enabled else "Offline"
                 status_line = f"│ Voice System: {status_text:<29} │"
                 startup_messages.insert(weapons_line + 1, status_line)
            
            # Find where to insert commands
             commands_line = next((i for i, line in enumerate(startup_messages) if "Commands:" in line), -1)
             if commands_line != -1 and self.voice_enabled:
                 startup_messages.insert(commands_line + 2, "  - 'voice' to toggle voice mode (text+voice/text only/voice only)")
                 startup_messages.insert(commands_line + 3, "  - 'tts' to switch to Windows TTS (if voice is garbled)")
        except Exception:
            pass # Fallback if config changed
        
        for line in startup_messages:
            if line is None: continue
            print(line)
            time.sleep(0.1)
        
        # Test voice on startup if available
        if self.voice_enabled:
            try:
                print("Voice System Test: ", end="", flush=True)
                self.bt_voice.speak("Voice systems operational, Pilot.", blocking=True)
                print("Complete.")
            except Exception as e:
                print(f"Failed ({e})")
    
    def display_shutdown(self):
        """Display BT-7274 shutdown sequence"""
        for line in config.SHUTDOWN_MESSAGES:
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
                    
                    # Check for TTS mode switch command
                    if user_input.lower() == "tts":
                        if self.voice_enabled:
                            self.bt_voice.force_windows_tts()
                            print("Switched to Windows TTS mode")
                        else:
                            print("Voice system not available")
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
