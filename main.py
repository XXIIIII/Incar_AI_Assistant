"""
Incar AI Assistant - Command Line Interface

A simplified CLI version for testing the core voice interaction features:
- Voice recording and speech recognition
- AI-powered travel assistance using LangChain and LLM
- Text-to-speech response generation

This serves as a lightweight alternative to the full GUI application.
"""

import os
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play

from Tutor.tutor import Tutor

# Constants
AUDIO_FREQ = 44100
RECORD_DURATION = 5
TEMP_AUDIO_PATH = 'tmp.wav'
RECORD_INPUT_PATH = 'record.wav'


def record_audio():
    """Record audio from microphone for specified duration."""
    recording = sd.rec(
        int(RECORD_DURATION * AUDIO_FREQ), 
        samplerate=AUDIO_FREQ, 
        channels=2
    )
    sd.wait()
    
    # Save and convert audio format
    write(RECORD_INPUT_PATH, AUDIO_FREQ, recording)
    audio = AudioSegment.from_wav(RECORD_INPUT_PATH)
    audio.export(TEMP_AUDIO_PATH, format="wav")
    
    return TEMP_AUDIO_PATH


def speech_to_text(audio_path):
    """Convert recorded audio to text using speech recognition."""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            return text
    except sr.UnknownValueError:
        print('Could not understand audio')
        return ""
    except sr.RequestError as e:
        print(f'Speech recognition error: {e}')
        return ""


def generate_ai_response(user_input):
    """Generate AI response using the Tutor agent."""
    
    try:
        agent = Tutor()
        response = agent.invoke(user_input)
        print(f'AI Response: "{response["speech"]}"')
        return response
    except Exception as e:
        print(f'AI processing error: {e}')
        return None


def text_to_speech(text, output_path=TEMP_AUDIO_PATH):
    """Convert text to speech and play it."""
    try:
        tts = gTTS(text=text, lang="zh", slow=False)
        tts.save(output_path)
        
        # Play the audio
        audio = AudioSegment.from_file(output_path)
        play(audio)
        
    except Exception as e:
        print(f'TTS error: {e}')


def cleanup_temp_files():
    """Remove temporary audio files."""
    temp_files = [RECORD_INPUT_PATH, TEMP_AUDIO_PATH]
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)

def main():
    """Main function for CLI voice interaction."""
    
    dotenv_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(dotenv_path, override=True)
    
    try:
        # Voice interaction pipeline
        audio_path = record_audio()
        user_text = speech_to_text(audio_path)
        
        if user_text:
            ai_response = generate_ai_response(user_text)
            
            if ai_response and ai_response.get('speech'):
                text_to_speech(ai_response['speech'])
                
                # Save travel schedule if available
                if ai_response.get('table'):
                    with open('schedule.md', 'w', encoding='utf-8') as f:
                        f.write(ai_response['table'])
                    print('Travel schedule saved to schedule.md')
            else:
                print('No valid AI response generated')
        else:
            print('No speech detected')
            
    except KeyboardInterrupt:
        print('\n Exiting...')
    except Exception as e:
        print(f'Unexpected error: {e}')
    finally:
        cleanup_temp_files()
        print('Session ended')


if __name__ == '__main__':
    main()