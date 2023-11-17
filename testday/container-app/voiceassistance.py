# General packages
import json
import os
import subprocess
from typing import List, Tuple, TypedDict, cast

import numpy as np
import requests
from scipy.io import wavfile

# available languages
available_lang = ["en", "nl", "fr", "de"]

# read fail messages, start messages and questions
fail_messages = {}
with open("chatbot_configs/fail_messages.txt") as f:
    for line in f:
        (key, val) = line.split(":", 1)
        fail_messages[key.strip()] = val.strip()

start_messages = {}
with open("chatbot_configs/start_messages.txt") as f:
    for line in f:
        (key, val) = line.split(":", 1)
        start_messages[key.strip()] = val.strip()

audio_files = {}
for lang in available_lang:
    audio_files[lang] = f"audio/start_message_{lang}.wav"

questions = {}
for lang in available_lang:
    with open(f"chatbot_configs/questions_{lang}.txt") as f:
        questions[lang] = f.readlines()

# configuration for Guardrails
with open("chatbot_configs/questionnaire.rail") as f:
    rail_string = f.read().rstrip()

# model endpoint information
url = "https://nlp-onwheels-end.westeurope.inference.ml.azure.com/score"
api_key = os.getenv("NLP_END_KEY") or ""
headers = {
    "Content-Type": "application/json",
    "Authorization": ("Bearer " + api_key),
    "azureml-model-deployment": "nlpmodels-1",
}


class ChatbotResponseDict(TypedDict):
    sample_rate: int
    audio: List
    memory: str
    chat_history: List[Tuple[str | None, str | None]]


class ResponseDict(TypedDict):
    text: ChatbotResponseDict


def voice_assistant(
    audio_file: str,
    language: str | None,
    chat_history: List[Tuple[str | None, str | None]],
    memory: str = "",
    guess_lang_flag: bool = False,
    audio_response_path: str = "response.wav",
) -> Tuple[str | None, list[Tuple[str | None, str | None]], str]:
    """Function to call the model endpoint for the voice assistant. It encapselates all
    the relevant information into a dictionary to send it to the model endpoint and
    post-processes the output of the model endpoint.

    Input:
        - audio_file            : the relative path to the audio file
        - language              : the preferd language
        - chat_history          : a list of all messages send between
                                  the user and the voice assistant
        - memory                : the variables to remember
        - guess_flag            : wheter or not to guess the language form the audio
        - audio_response_path   : the relative path to write the response to
    """
    chat_history = chat_history or [(None, None)]
    # set the language
    languages = {
        "English": "en",
        "Nederlands": "nl",
        "Francais": "fr",
        "Deutsch": "de",
        "Guess...": None,
    }
    language = "Guess..." if language not in languages.keys() else language
    language = languages[language]  # type: ignore
    guess_lang_flag = True if language is None else False

    # transform the audio file to a numpy array
    sr = 16000
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # Construct the data to be send to the model endpoint
    data = {
        "audio": audio.tolist(),
        "language": language,
        "guess_lang_flag": guess_lang_flag,
        "chat_history": chat_history,
        "memory": memory,
        "state": {
            "fail_message": fail_messages,
            "start_message": start_messages,
            "questions": questions,
            "rail_string": rail_string,
        },
        "openai_key": os.getenv("OPENAI_API_KEY"),
    }
    r = requests.post(url, data=json.dumps(data), headers=headers)
    response_dict = cast(ResponseDict, r.json())
    # retrieve the response from the model endpoint
    try:
        resp = response_dict["text"]

        # transform the audio numpy array to an audio file
        wav = np.array(resp["audio"])
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wavfile.write(
            audio_response_path, int(resp["sample_rate"]), wav_norm.astype(np.int16)
        )
        return audio_response_path, resp["chat_history"], resp["memory"]
    except Exception as e:
        print("Error: something went wrong in the maskrcnn endpoint")
        print(e)
        return None, chat_history, memory
