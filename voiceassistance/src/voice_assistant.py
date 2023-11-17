# General packages
import json
import os
from typing import Dict, List, Tuple

import dotenv
import flatdict

# text processing
import guardrails as gd
import openai
import torch
import whisper
from src.config import Config
from TTS.api import TTS
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

dotenv.load_dotenv()

config = Config()
os.makedirs(config.audio_dir, exist_ok=True)

# Configure the azure openai API
openai.api_type = config.openai_api_type
openai.api_base = config.openai_api_base
openai.api_version = config.openai_api_version

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading questionnaire
guard = gd.Guard.from_rail(config.railfile_path)
output_template = guard.output_schema.to_dict()

# Load all messages and questions
fail_messages = {}
with open(config.fail_messages_path) as f:
    for line in f:
        (key, val) = line.split(":", 1)
        fail_messages[key.strip()] = val.strip()

start_messages = {}
with open(config.start_messages_path) as f:
    for line in f:
        (key, val) = line.split(":", 1)
        start_messages[key.strip()] = val.strip()

questions = {}
for lang in config.lang:
    with open(config.questions_path[lang]) as f:
        questions[lang] = f.readlines()


def create_start_messages(tts: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create the audio files for the start messages (if they don't already exist)

    Input:
        tts : a dictionary of the corresponding TTS model for each available language

    Output:
        the audio and text version of the start messages
        (in dictionaries for every available language)
    """
    audio_files = {}
    for lang in config.lang:
        start_message_path = os.path.join(config.audio_dir, f"start_message_{lang}.wav")
        if not os.path.exists(start_message_path):
            audio_files[lang] = speak(
                tts=tts[lang], text=start_messages[lang], output_path=start_message_path
            )
        else:
            audio_files[lang] = start_message_path
    return audio_files, start_messages


def load_models(
    language: str | None = None,
) -> Tuple[whisper.model.Whisper, Dict[str, TTS]]:
    """Load the relevant models for STT and TTS. Option to only load for one language or
    for all available languages.

    Input:
        language : preferd language for the TTS

    Output:
        the whisper model and relevant TTS model(s)
    """
    lang_list = [language] if language is not None else config.lang
    os.makedirs(config.whisper_dir, exist_ok=True)
    whisper_model = whisper.load_model("medium", download_root=config.whisper_dir)
    gpu = torch.cuda.is_available()

    tts_models = dict()
    for language in lang_list:
        if not os.path.exists(
            config.tts_model_paths[language]["model_path"]  # type: ignore
        ):
            tts_models[language] = TTS(progress_bar=False, gpu=gpu)
            tts_models[language].manager = ModelManager(
                models_file=TTS.get_models_file_path(),
                output_prefix=config.models_path,
                progress_bar=False,
                verbose=False,
            )
            tts_models[language].load_tts_model_by_name(
                config.tts_model_paths[language]["model_name"], gpu
            )
        else:
            model_path = config.tts_model_paths[language]["model_path"]
            config_path = config.tts_model_paths[language]["config_path"]
            vocoder_path = config.tts_model_paths[language]["vocoder_path"]
            vocoder_config_path = config.tts_model_paths[language][
                "vocoder_config_path"
            ]
            tts_speakers_file = config.tts_model_paths[language]["tts_speakers_file"]
            tts_languages_file = config.tts_model_paths[language]["tts_languages_file"]

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

    return whisper_model, tts_models


def transcribe(
    audio_path: str,
    language: str,
    guess_flag: bool,
    whisper_model: whisper.model.Whisper,
) -> Tuple[str, str]:
    """Listen to the supplied audio with whisper and transcribe the text that is
    understood. If the language is not given by the user, the model will guess the
    language (from the availible options).

    Input:
        - audio_path    : filepath to the audio file that contains the user input
        - language      : the language to transcribe the audio in
        - guess_flag    : a boolean indicating if the language should be guessed
        - whisper_model : the whisper model to use for transcription

    Output:
        - text      : the transcribed text
        - language  : the language that was used to transcribe
    """

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    if guess_flag:
        print("Whisper: Guess language...")
        _, probs = whisper_model.detect_language(mel)
        probs = dict((k, probs[k]) for k in config.lang)
        language = max(probs, key=probs.get)
        print(f"Got language: {language}")

    print("Whisper: start transcribing...")
    options = whisper.DecodingOptions(language=language, fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    print("Whisper transcribed:", result.text)
    return result.text, language


def speak(
    tts: TTS, text: str = "Thank you for talking to me!", output_path: str | None = None
) -> str:
    """A wrapper function to use the text-to-speech model.

    Input:
        tts         : the relevant TTS model to use
        text        : the text (in string format) to translate to audio
        output_path : the relative path to write the audio to

    Output:
        audio_file : the resulting audio file
    """
    if output_path is None:
        output_path = os.path.join(config.audio_dir, "output.wav")
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path


def ask_questions(
    questions: List[str],
    fail_message: str,
    tts: TTS,
    max_questions: int = 3,
) -> Tuple[str, str]:
    """Create an audio file where the top questions are asked.

    Input:
        - questions     : a list of all (relevant) questions
        - fail_message  : fail message to revert to if there are no questions
        - language      : the language to ask the questions in
        - tts           : the TTS model
        - max_questions : the maximum of questions to ask

    Output:
        An audio file and the questiona that were asked
    """
    print("Coqui TTS: ask questions...")
    if len(questions):
        text = (
            " ".join(questions[: min(len(questions) - 1, max_questions)])
            if len(questions) > 1
            else questions[0]
        )
    else:
        text = fail_message
    audio_file = speak(tts=tts, text=text)
    return audio_file, text


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
            if choice is not None:
                info_choice = next(
                    iter(
                        input_dict[k]
                        .children.__dict__[choice]
                        .children.__dict__.values()
                    )
                ).children.__dict__
                variable_dict[choice] = create_variable_dict(info_choice, memory)
            else:
                variable_dict[choice] = choice
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


def extract_info(user_input: str, memory: str) -> Tuple[str | None, Dict | None]:
    """Extract the relevant info from the user input by utilising a LLM (gpt) and
    guardrails.

    Input:
        - user_input : as string containing the text that the user provided
        - memory     : the memory of the LLM (from previous inputs from the user)

    Output:
        - validated output : a JSON that fits all requirements
        - raw_llm_output   : the raw output of the LLM
    """
    # Wrap the OpenAI API call with the `guard` object
    try:
        raw_llm_output, validated_output = guard(
            openai.ChatCompletion.create,
            prompt_params={"user_input": user_input, "memory": memory},
            # engine="chatgpt",
            model="gpt-3.5-turbo",
            max_tokens=1024,
            temperature=0.1,
        )
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
            if v == "None" or v == "" or v == "{}" or v == "Null":
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
    lang: str,
    questions: List[str],
    memory: str = "",
    output_template: gd.schema.JsonSchema = output_template,
) -> Tuple[List[str], str]:
    """Process all info from the user to extract the necessary information, construct a
    list of relevant questions and construct the memory to remember for the next
    (conversation) loop.

    Input:
        - text              : the user input
        - lang              : the preferd language
        - memory            : the variables to remember
        - output_template   : the output template of the questionnaire
        - questions         : all questions of the questionnaire

    Output:
        - relevant_questions    : the relevant questions to ask for the next round
        - memory                : the relevant information to remember
    """

    # language model
    print("LLM: extract info...")
    raw_llm_output, llm_output = extract_info(text, memory)
    if llm_output is None:
        print("LLM raw output: ", raw_llm_output)
        return [fail_messages[lang]], memory

    # post processing
    output_json = post_process_json(llm_output)
    memory_dict = {k.split(".")[-1]: v for (k, v) in output_json.items()}
    variable_dict = create_variable_dict(output_template, memory_dict)
    var_ques_dict = fill_variable_dict_with_questions(variable_dict, questions)
    relevant_questions = find_missing_info(output_json, variable_dict, var_ques_dict)
    memory = str(memory_dict)
    return relevant_questions, memory


def talk(
    audio_path: str,
    lang: str,
    whisper_model: whisper.model.Whisper,
    tts_models: Dict[str, TTS],
    chat_history: List[Tuple[str | None, str | None]] | None = None,
    memory: str = "",
) -> Tuple[str, str, List[Tuple[str | None, str | None]], str]:
    """Transcribe the input audio file from the user, extract the needed information and
    speak back a response with relevant questions.

    Input:
        - audio_path    : the filepath of the audio file with the user input
        - lang          : the language that the user selected
        - chat_history  : a list of messages between the voice assistant and user
        - memory        : the information from the previous questions
        - whisper_model : the whisper model to use for STT
        - tts_models    : the TTS models for all available languages

    Output:
        - audio_file    : the audio response of the voice assistant
        - chat_history  : the updated messages list
        - memory        : the updates list of variables to remember
    """
    languages = config.languages
    lang = "Guess..." if lang not in languages.keys() else lang
    guess_flag = (
        True if (lang == "Guess..." and len(chat_history or []) <= 1) else False
    )

    text, language = transcribe(
        audio_path,
        languages[lang] or "en",
        guess_flag=guess_flag,
        whisper_model=whisper_model,
    )

    chat_history = chat_history or [(None, start_messages[language])]

    if text is not None:
        relevant_questions, memory = process_info(
            text, language, questions=questions[language], memory=memory
        )
        chat_history.append((text, memory))
    else:
        relevant_questions = []

    audio_file, asked_questions = ask_questions(
        relevant_questions,
        fail_message=fail_messages[language],
        tts=tts_models[language],
    )
    chat_history.append((None, asked_questions))

    return audio_file, language, chat_history, memory
