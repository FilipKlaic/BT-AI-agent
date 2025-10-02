#!/usr/bin/env python3
"""
Debug script for BT-7274 voice synthesis
"""

from bt_voice import BT7274Voice

def test_voice_debug():
    """Test voice synthesis with debug output"""
    try:
        print("Initializing BT-7274 voice system...")
        bt_voice = BT7274Voice()
        
        print(f"Sample rate: {bt_voice.sample_rate}")
        print(f"Available phonemes: {len(bt_voice.phoneme_id_map)}")
        
        # Test with very simple text
        test_texts = [
            "hello",
            "pilot", 
            "bt",
            "hello pilot"
        ]
        
        for text in test_texts:
            print(f"\n--- Testing: '{text}' ---")
            
            # Show phoneme conversion
            phonemes = bt_voice.text_to_phonemes(text)
            print(f"Text: '{text}' -> Phonemes: '{phonemes}'")
            
            # Show phoneme IDs
            phoneme_ids = bt_voice.phonemes_to_ids(phonemes)
            print(f"Phoneme IDs: {phoneme_ids}")
            
            # Test synthesis
            audio = bt_voice.synthesize_audio(text)
            print(f"Generated audio length: {len(audio)} samples")
            
            if len(audio) > 0:
                print(f"Audio min/max: {audio.min():.3f} / {audio.max():.3f}")
                
                # Save for inspection
                bt_voice.save_audio(audio, f"debug_{text.replace(' ', '_')}.wav")
                print(f"Saved as: debug_{text.replace(' ', '_')}.wav")
                
                # Play if requested
                play = input(f"Play audio for '{text}'? (y/n): ").lower().startswith('y')
                if play:
                    bt_voice.play_audio(audio, blocking=True)
            else:
                print("No audio generated!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_voice_debug()