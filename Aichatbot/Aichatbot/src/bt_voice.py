import onnxruntime
import numpy as np
import soundfile as sf
import sounddevice as sd
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional
import subprocess
import re

class BT7274Voice:
    def __init__(self, model_dir: str = None):
        """Initialize BT-7274's voice synthesis system"""
        if model_dir is None:
            # Default to voice_model directory relative to this file
            current_dir = Path(__file__).parent
            model_dir = current_dir.parent / "voice_model"
        
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "BT7274.onnx"
        self.config_path = self.model_dir / "BT7274.onnx.json"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        
        # Load model configuration
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Initialize ONNX runtime session
        self.session = onnxruntime.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']  # Use CPU for reliability
        )
        
        # Get model input/output info
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        
        print(f"Model inputs: {[inp.name for inp in self.inputs]}")
        print(f"Model outputs: {[out.name for out in self.outputs]}")
        
        # Main input is typically the first one (phoneme IDs)
        self.input_name = self.inputs[0].name
        self.output_name = self.outputs[0].name
        
        # Audio configuration
        self.sample_rate = self.config["audio"]["sample_rate"]
        self.phoneme_id_map = self.config["phoneme_id_map"]
        
        # Inference parameters
        self.noise_scale = self.config["inference"]["noise_scale"]
        self.length_scale = self.config["inference"]["length_scale"] 
        self.noise_w = self.config["inference"]["noise_w"]
        
        print(f"BT-7274 Voice System initialized. Sample rate: {self.sample_rate}Hz")
    
    def text_to_phonemes(self, text: str) -> str:
        """Convert text to phonemes using espeak"""
        try:
            # Use espeak to convert text to phonemes
            cmd = [
                'espeak-ng',
                '-q',  # quiet
                '--ipa',  # IPA phonemes
                '-v', 'en',  # English voice
                text
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                phonemes = result.stdout.strip()
                # Clean up the phonemes
                phonemes = re.sub(r'[ˈˌ]', '', phonemes)  # Remove stress marks
                return phonemes
            else:
                print(f"Warning: espeak failed, using fallback for: {text}")
                return self._fallback_phonemes(text)
                
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Warning: espeak not available, using fallback for: {text}")
            return self._fallback_phonemes(text)
    
    def _fallback_phonemes(self, text: str) -> str:
        """Fallback phoneme conversion using English-like phoneme patterns"""
        text = text.lower().strip()
        
        # Replace common letter combinations with IPA-like phonemes that exist in our map
        # This should sound more English-like with Piper models
        replacements = [
            # Vowel combinations
            ('ou', 'aʊ'), ('ow', 'aʊ'), ('oy', 'ɔɪ'), ('oi', 'ɔɪ'),
            ('ay', 'eɪ'), ('ai', 'eɪ'), ('ei', 'eɪ'),
            ('ee', 'i'), ('ea', 'i'), ('ie', 'i'),
            ('oo', 'u'), ('ou', 'u'),
            
            # Consonant combinations  
            ('th', 'θ'), ('sh', 'ʃ'), ('ch', 'tʃ'), ('ng', 'ŋ'),
            ('ph', 'f'), ('gh', 'f'), ('ck', 'k'), ('qu', 'kw'),
            
            # Single vowels to more English-like sounds
            ('a', 'æ'), ('e', 'ɛ'), ('i', 'ɪ'), ('o', 'ɔ'), ('u', 'ʌ'),
        ]
        
        # Apply replacements in order
        for old, new in replacements:
            text = text.replace(old, new)
        
        # Clean result to only include available phonemes
        result = ""
        i = 0
        while i < len(text):
            char = text[i]
            
            # Check for 2-character phonemes first
            if i < len(text) - 1:
                two_char = text[i:i+2]
                if two_char in self.phoneme_id_map:
                    result += two_char
                    i += 2
                    continue
            
            # Single character phonemes
            if char in self.phoneme_id_map:
                result += char
            elif char.isspace():
                result += " "
            elif char in ".,!?":
                if char in self.phoneme_id_map:
                    result += char
            # Skip unknown characters
            
            i += 1
        
        return result
    
    def phonemes_to_ids(self, phonemes: str) -> list:
        """Convert phonemes to phoneme IDs"""
        phoneme_ids = []
        
        # Add beginning of sequence
        phoneme_ids.append(self.phoneme_id_map["^"][0])
        
        # Convert each character/phoneme - handle multi-character phonemes
        i = 0
        while i < len(phonemes):
            char = phonemes[i]
            
            # Check for multi-character phonemes first
            if i < len(phonemes) - 1:
                two_char = phonemes[i:i+2]
                if two_char in self.phoneme_id_map:
                    phoneme_ids.append(self.phoneme_id_map[two_char][0])
                    i += 2
                    continue
            
            # Single character phonemes
            if char in self.phoneme_id_map:
                phoneme_ids.append(self.phoneme_id_map[char][0])
            elif char == ' ':
                phoneme_ids.append(self.phoneme_id_map[" "][0])
            elif char in '.,!?:;':
                # Add pauses for punctuation
                if char in self.phoneme_id_map:
                    phoneme_ids.append(self.phoneme_id_map[char][0])
                else:
                    phoneme_ids.append(self.phoneme_id_map[" "][0])  # Use space as pause
            else:
                # Skip unknown characters but don't spam warnings for fallback mode
                pass
            
            i += 1
        
        # Add end of sequence
        phoneme_ids.append(self.phoneme_id_map["_"][0])
        
        return phoneme_ids
    
    def synthesize_audio(self, text: str) -> np.ndarray:
        """Convert text to audio using the BT-7274 voice model"""
        if not text or len(text.strip()) == 0:
            return np.array([])
            
        # Convert text to phonemes
        phonemes = self.text_to_phonemes(text)
        
        # Convert phonemes to IDs
        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        if len(phoneme_ids) <= 2:  # Only start/end tokens
            return np.array([])
        
        # Validate phoneme IDs are in valid range
        max_id = max(max(ids) for ids in self.phoneme_id_map.values())
        phoneme_ids = [pid for pid in phoneme_ids if 0 <= pid <= max_id]
        
        if len(phoneme_ids) <= 2:
            print(f"Warning: No valid phoneme IDs generated for text: '{text}'")
            return np.array([])
        
        # Prepare inputs for ONNX model
        phoneme_ids_array = np.array(phoneme_ids, dtype=np.int64).reshape(1, -1)
        input_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
        
        # Prepare scales (noise_scale, length_scale, noise_w)
        scales = np.array([self.noise_scale, self.length_scale, self.noise_w], dtype=np.float32)
        
        # Build input dictionary based on what the model expects
        input_dict = {
            "input": phoneme_ids_array,
            "input_lengths": input_lengths,
            "scales": scales
        }
        
        # Run inference
        try:
            outputs = self.session.run(None, input_dict)
            audio = outputs[0]
            
            # The output is typically in the shape (1, audio_length)
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            
            # Normalize audio to prevent clipping
            if len(audio) > 0:
                max_val = np.abs(audio).max()
                if max_val > 0:
                    audio = audio / max_val * 0.8  # Scale to 80% to prevent clipping
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"Voice synthesis error: {e}")
            print(f"Phoneme IDs: {phoneme_ids[:10]}..." if len(phoneme_ids) > 10 else f"Phoneme IDs: {phoneme_ids}")
            print(f"Input shapes: input={phoneme_ids_array.shape}, lengths={input_lengths.shape}, scales={scales.shape}")
            return np.array([])
    
    def play_audio(self, audio: np.ndarray, blocking: bool = False):
        """Play audio using sounddevice"""
        if len(audio) == 0:
            print("Warning: Empty audio array, nothing to play")
            return
        
        try:
            if blocking:
                sd.play(audio, samplerate=self.sample_rate)
                sd.wait()  # Wait until the audio finishes
            else:
                # Play in a separate thread for non-blocking playback
                def play_thread():
                    sd.play(audio, samplerate=self.sample_rate)
                    sd.wait()
                
                thread = threading.Thread(target=play_thread, daemon=True)
                thread.start()
                
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def save_audio(self, audio: np.ndarray, filename: str):
        """Save audio to a file"""
        try:
            sf.write(filename, audio, self.sample_rate)
            print(f"Audio saved to: {filename}")
        except Exception as e:
            print(f"Error saving audio: {e}")
    
    def speak(self, text: str, save_to_file: Optional[str] = None, blocking: bool = False):
        """Convert text to speech and play it (BT-7274's main voice function)"""
        try:
            # Limit text length for better synthesis
            if len(text) > 100:
                text = text[:100].rsplit(' ', 1)[0] + "."
            
            # Generate audio
            audio = self.synthesize_audio(text)
            
            if len(audio) == 0:
                print(f"Warning: No audio generated for: '{text[:30]}...'")
                return
            
            # Save if requested
            if save_to_file:
                self.save_audio(audio, save_to_file)
            
            # Play the audio
            self.play_audio(audio, blocking=blocking)
            
        except Exception as e:
            print(f"Voice Error: {e}")
    
    def test_voice(self):
        """Test the voice system with a simple phrase"""
        test_phrase = "Hello pilot."
        print(f"Testing voice with: '{test_phrase}'")
        
        # Debug phoneme conversion
        phonemes = self.text_to_phonemes(test_phrase)
        phoneme_ids = self.phonemes_to_ids(phonemes)
        print(f"Phonemes: '{phonemes}'")
        print(f"Phoneme IDs: {phoneme_ids}")
        
        self.speak(test_phrase, blocking=True)

if __name__ == "__main__":
    # Test the voice system
    try:
        bt_voice = BT7274Voice()
        bt_voice.test_voice()
    except Exception as e:
        print(f"Error initializing BT-7274 voice: {e}")