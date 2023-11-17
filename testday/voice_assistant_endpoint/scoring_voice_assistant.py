# General packages
import json
import logging
import os
from typing import Dict, List, Tuple

import flatdict

# text processing
import guardrails as gd
import numpy as np
import openai
import torch
import whisper
from TTS.api import TTS
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.utils.synthesizer import Synthesizer

# see if GPU is available
device = torch.cuda.is_available()

# Openai configuration
openai.api_type = "azure"
openai.api_base = "https://openai-onwheels.openai.azure.com/"
openai.api_version = "2023-03-15-preview"

# available languages
available_lang = ["en", "nl", "fr", "de"]

whisper_model: whisper.model.Whisper
tts_models: Dict[str, TTS]


def init() -> None:
    """This function is called when the container is initialized/started, typically
    after create/update of the deployment.

    The logic is written here to perform init operations like caching the models in
    memory
    """
    global whisper_model, tts_models

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    models_path = os.path.join(os.getenv("AZUREML_MODEL_DIR") or ".", "NLPmodels")

    whisper_model = whisper.load_model("medium", download_root=models_path)

    model_files_TTS = {
        "en": {
            "model_path": "tts/tts_models--en--ljspeech--vits/model_file.pth",
            "config_path": "tts/tts_models--en--ljspeech--vits/config.json",
            "vocoder_path": None,
            "vocoder_config_path": None,
            "tts_speakers_file": None,
            "tts_languages_file": None,
        },
        "nl": {
            "model_path": "tts/tts_models--nl--css10--vits/model_file.pth.tar",
            "config_path": "tts/tts_models--nl--css10--vits/config.json",
            "vocoder_path": None,
            "vocoder_config_path": None,
            "tts_speakers_file": "tts/tts_models--nl--css10--vits/speaker_ids.json",
            "tts_languages_file": "tts/tts_models--nl--css10--vits/language_ids.json",
        },
        "fr": {
            "model_path": "tts/tts_models--fr--css10--vits/model_file.pth.tar",
            "config_path": "tts/tts_models--fr--css10--vits/config.json",
            "vocoder_path": None,
            "vocoder_config_path": None,
            "tts_speakers_file": "tts/tts_models--fr--css10--vits/speaker_ids.json",
            "tts_languages_file": "tts/tts_models--fr--css10--vits/language_ids.json",
        },
        "de": {
            "model_path": "tts/tts_models--de--thorsten--vits/model_file.pth",
            "config_path": "tts/tts_models--de--thorsten--vits/config.json",
            "vocoder_path": None,
            "vocoder_config_path": None,
            "tts_speakers_file": None,
            "tts_languages_file": None,
        },
    }

    tts_models = dict()
    for language in available_lang:
        model_path = None
        config_path = None
        vocoder_path = None
        vocoder_config_path = None
        tts_speakers_file = None
        tts_languages_file = None

        if model_files_TTS[language]["model_path"]:
            model_path = os.path.join(
                models_path, model_files_TTS[language]["model_path"]  # type: ignore
            )
        if model_files_TTS[language]["config_path"]:
            config_path = os.path.join(
                models_path, model_files_TTS[language]["config_path"]  # type: ignore
            )
        if model_files_TTS[language]["vocoder_path"]:
            vocoder_path = os.path.join(
                models_path, model_files_TTS[language]["vocoder_path"]  # type: ignore
            )
        if model_files_TTS[language]["vocoder_config_path"]:
            vocoder_config_path = os.path.join(
                models_path,
                model_files_TTS[language]["vocoder_config_path"],  # type: ignore
            )
        if model_files_TTS[language]["tts_speakers_file"]:
            tts_speakers_file = os.path.join(
                models_path,
                model_files_TTS[language]["tts_speakers_file"],  # type: ignore
            )
        if model_files_TTS[language]["tts_languages_file"]:
            tts_languages_file = os.path.join(
                models_path,
                model_files_TTS[language]["tts_languages_file"],  # type: ignore
            )

        tts_models[language] = TTS(progress_bar=False, gpu=device)
        tts_models[language].synthesizer = Synthesizer(
            tts_checkpoint=None,
            tts_config_path=config_path,
            tts_speakers_file=tts_speakers_file,
            tts_languages_file=tts_languages_file,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config_path,
            encoder_checkpoint=None,
            encoder_config=None,
            use_cuda=torch.cuda.is_available(),
        )
        # force the package to read in the correct speaker file
        tts_models[language].synthesizer.tts_config = load_config(config_path)
        tts_models[
            language
        ].synthesizer.tts_config.model_args.speakers_file = tts_speakers_file
        tts_models[language].synthesizer.output_sample_rate = tts_models[
            language
        ].synthesizer.tts_config.audio["sample_rate"]
        tts_models[language].synthesizer.tts_model = setup_tts_model(
            config=tts_models[language].synthesizer.tts_config
        )
        tts_models[language].synthesizer.tts_model.load_checkpoint(
            tts_models[language].synthesizer.tts_config, model_path, eval=True
        )
        if torch.cuda.is_available():
            tts_models[language].synthesizer.tts_model.cuda()

    logging.info("Init complete")


def speak(text: str, language: str) -> Tuple[np.ndarray, int]:
    """Convert the text respons of the voice assistant to audio by using the TTS model
    for the preferd language.

    Input:
        - text      : text to convert to audio
        - language  : prefered language to be used for the TTS model

    Output:
        - wav : a numpy array for the audio
        - sr  : the sampling rate of the audio
    """
    logging.info("coquiTTS: request received")
    wav = tts_models[language].tts(text=text)  # noqa: F821
    sr = tts_models[language].synthesizer.output_sample_rate  # noqa: F821
    logging.info("coquiTTS: request processed")

    return wav, sr


def transcribe(
    audio: List,
    language: str,
    guess_flag: bool = False,
    available_lang: List[str] = available_lang,
) -> Tuple[str, str]:
    """A function to transcibe audio to text with the whisper model. This function can
    also guess the language of the spoken text.

    Input:
        - audio             : list with content of the audio file
        - language          : preferd language to transcribe into
        - guess_flag        : if the Whisper model should guess the language
        - avialable_lang    : the available languages to transcribe to

    Output:
        - the transcribed text
        - the infered language (if Whisper needed to guess)
    """
    logging.info("whisper: request received")
    audio_array = np.array(audio, dtype=np.float32)
    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array).to(
        whisper_model.device  # noqa: F821
    )
    logging.info("whisper: audio loaded")

    if guess_flag:
        _, probs = whisper_model.detect_language(mel)  # noqa: F821
        probs = dict((k, probs[k]) for k in available_lang)
        language = max(probs, key=probs.get)
        logging.info(f"Got language: {language}")

    options = whisper.DecodingOptions(language=language, fp16=False)
    result = whisper.decode(whisper_model, mel, options)  # noqa: F821
    logging.info(f"whisper transcribed: {result.text}")
    logging.info("whisper: request processed")

    return result.text, language


def ask_questions(
    questions: List[str], max_questions: int = 3, language: str = "en"
) -> Tuple[np.ndarray, str, int]:
    """Create an audio file where the top questions are asked.

    Input:
        - questions     : a list of all (relevant) questions
        - max_questions : the maximum of questions to ask
        - language      : the language to ask the questions in


    Output:
        - the questions in audio format
        - the question in text format
        - the sampling rate of the audio
    """
    assert len(questions), "An empty list was given to this function"
    text = (
        " ".join(questions[: min(len(questions) - 1, max_questions)])
        if len(questions) > 1
        else questions[0]
    )
    logging.info(f"Speaking: {text}")
    audio, sr = speak(text, language)
    return audio, text, sr


def create_variable_dict(input_dict: Dict, memory: Dict[str, str]) -> Dict:
    """Get a dictionary with all variables that are included in the rail file (in the
    same structure)

    Input:
        - input_dict : the output template of guardrails
        - memory     : the memory of the LLM with already answered variables
        (relevant for choices in the guardrails file)

    Output:
        The dictionary with all variables of the questionnaire
    """
    variable_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, (gd.datatypes.Choice)) and k in memory.keys():
            choice = memory[k]
            info_choice = next(
                iter(input_dict[k].children.__dict__[choice].children.__dict__.values())
            ).children.__dict__
            variable_dict[choice] = create_variable_dict(info_choice, memory)
        elif isinstance(
            v, (gd.datatypes.Case, gd.datatypes.Object, gd.datatypes.Choice)
        ):
            variable_dict[k] = create_variable_dict(
                input_dict[k].children.__dict__, memory
            )
        else:
            variable_dict[k] = ""  # type: ignore
    return variable_dict


def fill_variable_dict_with_questions(
    variable_dict: Dict, questions: List[str]
) -> Dict:
    """Link the corresponding questions to all variables of the questionnaire.

    Input:
        - variable_dict : a dictionary of all variables of the questionnaire
        - questions     : a list of all questions
        (beginning with the variable they correspond to)

    Output:
        a dictionary with the variables as keys and the questions as values
    """
    var_ques_dict = variable_dict.copy()
    question_dict = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in questions
    }
    for k, v in var_ques_dict.items():
        if isinstance(v, str) and k in question_dict.keys():
            var_ques_dict[k] = question_dict[k] if len(question_dict[k]) else ""
        if isinstance(v, dict):
            if len(v):
                var_ques_dict[k] = fill_variable_dict_with_questions(
                    variable_dict[k], questions
                )
            else:
                var_ques_dict[k] = ""
    return var_ques_dict


def extract_info(
    user_input: str, memory: str, guard: gd.guard.Guard
) -> Tuple[str | None, Dict | None]:
    """Extract the relevant info from the user input by utilising a LLM (gpt) and
    guardrails.

    Input:
        - user_input : as string containing the text that the user provided
        - memory     : the memory of the LLM (from previous inputs from the user)
        - guard      : the guardrail configuration

    Output:
        - validated_output  : the JSON that follows all requirements
        - raw_llm_output    : the raw output of the LLM
    """
    # Wrap the OpenAI API call with the `guard` object
    try:
        raw_llm_output, validated_output = guard(
            openai.ChatCompletion.create,
            prompt_params={"user_input": user_input, "memory": memory},
            engine="chatgpt",
            max_tokens=1024,
            temperature=0.1,
        )
        print(raw_llm_output, validated_output)
        if not isinstance(validated_output, dict):
            print("LLM: no valid output")
            validated_output = None
    except Exception as e:
        print("error in guardrails:", e)
        return None, None
    if validated_output is not None and "location_info" not in validated_output.keys():
        validated_output = {"location_info": validated_output}
    return raw_llm_output, validated_output


def post_process_json(llm_output: Dict) -> Dict:
    """Process the output json by evaluating some strings into the right format and
    flattening the json.

    Input:
        llm_output : the output of the LLM

    Output:
        the processed json
    """
    try:
        if isinstance(llm_output, str):
            llm_output = json.loads(llm_output)
        output_json = flatdict.FlatDict(llm_output, delimiter=".")
        output_json = {
            k: v
            for (k, v) in output_json.items()
            if not isinstance(v, flatdict.FlatDict)
        }
        for k, v in output_json.items():
            if v == "None" or v == "" or v == "{}":
                output_json[k] = None
            elif v == "False":
                output_json[k] = False
            elif v == "True":
                output_json[k] = True

        output_json = {k: v for (k, v) in output_json.items() if v is not None}
    except Exception as e:
        print("Error while post-processing the output of the LLM: ", e)
        return {}
    return output_json


def find_missing_info(
    output_json: Dict, variable_dict: Dict, var_ques_dict: Dict[str, str]
) -> List[str]:
    """Compare the output of the LLM to the output template to find the missing info and
    ask the relevant questions.

    Input:
        - output_json   : the (post-processed) json output of the LLM
        - variable_dict : a dictionairy with all variables from the questionnaire
        - var_ques_dict : the link between these variables and their questions

    Output:
        The relevant questions to ask
    """
    var_output = output_json.keys()
    var_template = dict(flatdict.FlatDict(variable_dict, delimiter=".")).keys()
    relevant_questions = [
        flatdict.FlatDict(var_ques_dict, delimiter=".")[k]
        for k in var_template
        if k not in var_output
    ]
    relevant_questions = [q for q in relevant_questions if len(q)]
    return relevant_questions


def process_info(
    text: str,
    guard: gd.guard.Guard,
    questions: List[str],
    memory: str = "",
    fail_message: str = "",
) -> Tuple[List[str], str]:
    """Process all info from the user to extract the necessary information, construct a
    list of relevant questions and construct the memory to remember for the next
    (conversation) loop.

    Input:
        - text              : the user input
        - memory            : the variables to remember
        - guard             : the guardrail configuration
        - questions         : all questions of the questionnaire
        - fail_message      : a fail message to fall back to

    Output:
        - relevant_questions    : the relevant questions to ask for the next round
        - memory                : the relevant information to remember
    """

    # language model
    raw_llm_output, llm_output = extract_info(text, memory, guard=guard)
    if llm_output is None or not llm_output and raw_llm_output is not None:
        logging.info("No validated output from LLM", json.dumps(raw_llm_output))
        return [fail_message], memory

    # post processing
    output_json = post_process_json(llm_output)
    memory_dict = {k.split(".")[-1]: v for (k, v) in output_json.items()}
    variable_dict = create_variable_dict(guard.output_schema, memory_dict)
    var_ques_dict = fill_variable_dict_with_questions(variable_dict, questions)
    relevant_questions = find_missing_info(output_json, variable_dict, var_ques_dict)
    memory = str(memory_dict)
    return relevant_questions, memory


def run(raw_data: str) -> Dict:
    """This function is exectuted everytime anybody calls the model endpoint. It reads
    the data and calls all necessary subfunctions. It returns either an error message or
    a succesful response of the voice assistant.

    Input:
        - raw data : a dictonairy with all necessary information in string format

    Output:
        a dictionary in string format with audio, sample rate, chat_history,
        memory and an optional error message
    """

    logging.info("voice_assistant: request received")
    try:
        data = json.loads(raw_data)
    except Exception as e:
        return {
            "audio": None,
            "sample_rate": None,
            "chat_history": None,
            "memory": None,
            "message": e,
        }

    needed_keys = [
        "audio",
        "language",
        "guess_lang_flag",
        "chat_history",
        "memory",
        "state",
        "openai_key",
    ]
    for key in needed_keys:
        if key not in data.keys():
            return {
                "audio": None,
                "sample_rate": None,
                "chat_history": None,
                "memory": None,
                "message": f"Request should contain \
                    the following parameters: {needed_keys}",
            }

    # set openai api key (TODO: do this more safe)
    openai.api_key = data["openai_key"]

    # Loading questionnaire
    guard = gd.Guard.from_rail_string(data["state"]["rail_string"])

    if data["language"] not in available_lang and data["language"] is not None:
        return {
            "audio": None,
            "sample_rate": None,
            "chat_history": None,
            "memory": None,
            "message": f"Language should be one of the following: {available_lang}",
        }

    # STT
    text, language = transcribe(
        data["audio"], data["language"], guess_flag=data["guess_lang_flag"]
    )

    chat_history = data["chat_history"] or [
        (None, data["state"]["start_message"][language])
    ]

    # NLP
    relevant_questions, memory = process_info(
        text,
        guard=guard,
        questions=data["state"]["questions"][language],
        memory=data["memory"],
        fail_message=data["state"]["fail_message"][language],
    )
    chat_history.append((text, memory))

    # TTS
    audio, asked_questions, sr = ask_questions(relevant_questions, language=language)
    audio_list = [float(number) for number in audio]
    chat_history.append((None, asked_questions))

    return {
        "audio": audio_list,
        "sample_rate": str(sr),
        "chat_history": chat_history,
        "memory": memory,
        "state": data["state"],
        "message": "Successfully responded",
    }
