import os
from enum import Enum
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from src.voice_assistant import load_models, start_messages, talk

app = FastAPI()


class Language(str, Enum):
    Dutch = "Nederlands"
    English = "English"
    French = "Francais"
    German = "Deutsch"


# load the models
whisper_model, tts_models = load_models()


@app.post("/talk/")
async def voice_assistant(
    audio_file: UploadFile = File(...),
    language: Language = Language.English,
    memory: str = "",
) -> Any:
    """Function for post requests to talk to the voice assistant.

    Input:
        - audo_file : audio containing the speech of the user
        - language  : prefered language of the user
        - memory    : previous knowledge of the voice assistant

    Output:
        audio file with the response of the voice assistant
    """

    # save local copy if input
    try:
        contents = audio_file.file.read()
        with open(audio_file.filename, "wb") as f:  # type: ignore
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        audio_file.file.close()

    # welcomes message
    start_message = (
        start_messages[language.value]
        if language.value in start_messages.keys()
        else None
    )

    if audio_file.filename:
        # Talk to the voice assistant
        answer_file, _, _, memory = talk(
            audio_file.filename,
            language.value,
            whisper_model=whisper_model,
            tts_models=tts_models,
            chat_history=[(None, start_message)],
            memory=memory,
        )

        # Also save the variables to remember inside the header of the response
        results = {"memory": memory}

        # Return the audio response of the voice assistant
        return FileResponse(answer_file, media_type="audio/wav", headers=results)
    else:
        return {"message": "There was an error reading the file"}


@app.get("/listen/")
async def listen_to_response(answer_file: str = "output.wav") -> Any:
    """Get function to listen to the latest response of the voice assistant because
    Swagger UI doesn't return audio correctly in a post function.

    Input:
        answer_file : relative path to the answer file

    Output:
        playable audio file
    """
    if os.path.exists(answer_file):
        return FileResponse(answer_file, media_type="audio/wav")  # type: ignore
    else:
        return {"message": "There is no such file"}
