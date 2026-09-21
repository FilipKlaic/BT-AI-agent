"""
Piper TTS integration for BT-7274 voice model
This uses the proper Piper TTS system instead of manual ONNX inference
"""

import subprocess
import os
import tempfile
import threading
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional

class PiperBTVoice:
    def __init__(self, model_dir: str = None):
        """Initialize Piper TTS for BT-7274"""
        if model_dir is None:
            current_dir = Path(__file__).parent
            model_dir = current_dir.parent / "voice_model"
        
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "BT7274.onnx"
        self.config_path = self.model_dir / "BT7274.onnx.json"
        
        # Check if Piper is available
        self.piper_available = self._check_piper_installation()
        
        if not self.piper_available:
            raise RuntimeError("Piper TTS not found. Please install Piper TTS.")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        print("Piper BT-7274 Voice System initialized")
    
    def _check_piper_installation(self) -> bool:
        """Check if Piper is installed and accessible"""
        # Check for local binary first
        local_piper = Path(__file__).parent / "piper" / "piper"
        if local_piper.exists() and local_piper.is_file():
            self.piper_cmd = str(local_piper)
            return True
            
        # Check system PATH
        try:
            result = subprocess.run(['piper', '--help'], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=5)
            if result.returncode == 0:
                self.piper_cmd = 'piper'
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
            
        return False
    
    def synthesize_text_to_wav(self, text: str, output_file: str) -> bool:
        """Use Piper to synthesize text to WAV file"""
        try:
            cmd = [
                self.piper_cmd,
                '--model', str(self.model_path),
                '--config', str(self.config_path),
                '--output_file', output_file
            ]
            
            # Run piper with text input
            result = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(output_file):
                return True
            else:
                print(f"Piper error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Piper synthesis error: {e}")
            return False
    
    def play_wav_file(self, wav_file: str, blocking: bool = False):
        """Play WAV file using sounddevice"""
        try:
            data, samplerate = sf.read(wav_file)
            
            if blocking:
                sd.play(data, samplerate)
                sd.wait()
            else:
                def play_async():
                    sd.play(data, samplerate)
                    sd.wait()
                threading.Thread(target=play_async, daemon=True).start()
                
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def speak(self, text: str, blocking: bool = False):
        """Main speak function using Piper TTS"""
        try:
            # Limit text length
            if len(text) > 200:
                text = text[:200].rsplit(' ', 1)[0] + "."
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav = temp_file.name
            
            # Synthesize with Piper
            if self.synthesize_text_to_wav(text, temp_wav):
                # Play the generated audio
                self.play_wav_file(temp_wav, blocking)
                
                # Clean up temp file after a delay
                if not blocking:
                    def cleanup():
                        import time
                        time.sleep(10)  # Wait for audio to finish
                        try:
                            os.unlink(temp_wav)
                        except:
                            pass
                    threading.Thread(target=cleanup, daemon=True).start()
                else:
                    os.unlink(temp_wav)
            else:
                print("Piper synthesis failed")
                
        except Exception as e:
            print(f"Piper speak error: {e}")
    
    def test_voice(self):
        """Test the Piper voice system"""
        test_text = "Voice systems operational, Pilot. Protocol Three: Protect the Pilot."
        print(f"Testing Piper BT-7274 voice: '{test_text}'")
        self.speak(test_text, blocking=True)

# Installation check function
def check_piper_setup():
    """Check if Piper is properly set up"""
    try:
        voice = PiperBTVoice()
        print("Piper BT-7274 voice system ready!")
        return True
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTo install Piper TTS:")
        print("1. Download from: https://github.com/rhasspy/piper/releases")
        print("2. Extract piper.exe to your PATH or this directory")
        print("3. Or install via: pip install piper-tts")
        return False
    except Exception as e:
        print(f"Setup error: {e}")
        return False

if __name__ == "__main__":
    if check_piper_setup():
        voice = PiperBTVoice()
        voice.test_voice()