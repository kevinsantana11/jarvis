import openai
import wave
import pyaudio
import requests
import os

from typing import List
import dotenv

CHANNELS = 1
CHUNK = 1024
RATE = 44100
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 5

dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_API_ORG")

pyaudio_instance = pyaudio.PyAudio()

messages = [
    {"role": "system", "content": "You are an home assistant which converts natural language to system commands."},
    {"role": "system", "content": "The user will interact with you using different languages: English and Spanish"},
    {"role": "system", "content": "You will reply in the response_format: <entity_id>:<state>"},
    {"role": "system", "content": "entity_id: [room_light|cristinas_office_light|galaxy_projector]"},
    {"role": "system", "content": "state: [on|off]"},
    {"role": "system", "content": "ONLY respond using the response_format"},
    {"role": "system", "content": "ALWAYS respond using the response_format"},
    {"role": "system", "content": "ONLY respond using the response_format even when you don't know the language."},
]

def process_command(command: str):
    command_parts = command.split(":")
    command_entity = command_parts[0]
    command_entity_state = command_parts[1]
    home_automate(command_entity, command_entity_state)

def home_automate(entity_id, state):
    bearer_token = "Bearer {}".format(TOKEN)
    headers = {"authorization": bearer_token, "content-type": "application/json"}
    data = { "entity_id": "light.{}".format(entity_id)}
    url = "http://localhost:8123/api/services/light/turn_{}".format(state)
    response = requests.post(url, headers=headers, json=data)
    print(response)

def get_user_input(text_flag):
    if text_flag:
        return input("prompt: ")
    else:
        recording_file = wave.open("temp_recording.wav", "wb")
        pyaudio_instance.__init__() # Initialize pyaudio incase it wasn't

        recording_file.setnchannels(CHANNELS)
        recording_file.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
        recording_file.setframerate(RATE)
        
        stream = pyaudio_instance.open(RATE, CHANNELS, FORMAT, input=True)
        print("[recording-start]")
        for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
            recording_file.writeframes(stream.read(CHUNK))
        print("[recording-end]")

        stream.close()
        pyaudio_instance.terminate() # Terminate the pyaudio instance

        recording_file.close()
        recording_file = open("temp_recording.wav", "rb")

        audio_transcribe = openai.Audio.transcribe('whisper-1', recording_file)
        transcribed_audio = audio_transcribe.get("text")
        recording_file.close()

        print("[transcription] - {}".format(transcribed_audio))
        return transcribed_audio


def jarvis(messages: List[any], text_flag):
    while True:
        proceed = input("yes|no|_")
        
        if proceed != "yes":
            break

        user_input = get_user_input(text_flag)

        if user_input is None or user_input == "":
            continue

        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)

        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        top_response_message = chat_response.get("choices", [])[0].get("message")
        messages.append(top_response_message)
        output_message = "{} - {}".format(top_response_message.get("role"), top_response_message.get("content"))
        print(output_message)

        process_command(top_response_message.get("content"))

jarvis(messages, text_flag=False)