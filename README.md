# Jarvis
Simple project leveraging Whisper(Voice Audio -> Text), ChatGPT(Text -> commands) and my local running instance of Home-Assitant A.I to create a sort of voice interface home assistant. 

Interesting thing here is that it can convert commands from many different languages (English, Spanish, Japanese tested so far) into commands.

(this is just a test edit)

# Starting it up
0) Have the home-assistant.io server up and running
1) Make sure to set the following env-vars in an .env file
```
TOKEN=<HOME_ASSISTANT_TOKEN>
OPENAI_API_KEY=<OPENAI_API_KEY>
OPENAI_API_ORG=<OPEN_API_ORG>
```
2) Create a virtual environment: `python3 -m virtualenv .venv`
  a) activate it `source .venv/bin/activate`
3) `pip install -r requirements 
  a) You may have some issues installing pyaudio, to fix that on debian install the following packages `portaudio19-dev python3-all-dev` with your favorite package manager
4) `python jarvis.py`
