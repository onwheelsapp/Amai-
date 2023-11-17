# general imports
from typing import Dict, List, Tuple

import gradio as gr
from src.config import Config

# helper files
from src.voice_assistant import create_start_messages, load_models, talk
from TTS.api import TTS
from whisper.model import Whisper

config = Config()
# available languages
languages = config.languages

# load the models
whisper_model, tts_models = load_models()

# create the start messages if they don't already exist
audio_files, start_messages = create_start_messages(tts_models)


def greet(
    language: str, chatbot: List, audio_output: str
) -> Tuple[str, str, List[Tuple[str | None, str | None]]]:
    """Function to dynamicly change the welcome message based on the language the user
    chooses.

    Input:
        - language      : the preferd language
        - chatbot       : a list of messages with the chatbot
        - audio_output  : the welcome message

    Output:
        - language      : the prefered language
        - audio_output  : the corresponding welcome message (audio)
        - chatbot       : the corresponding welcome message (text)
    """
    if len(chatbot) == 1:
        audio_output = audio_files[languages[language] or "en"]
        chatbot = [(None, start_messages[languages[language] or "en"])]
    return language, audio_output, chatbot


def voice_assistant(
    audio: str,
    language: str,
    chatbot: List[Tuple[str | None, str | None]],
    memory: str,
    whisper_model: Whisper = whisper_model,
    tts_models: Dict[str, TTS] = tts_models,
) -> Tuple[str, str, List[Tuple[str | None, str | None]], str]:
    """Wrapper function to start talking to the voice assistant. This fixes the whisper
    and TTS models.

    Input:
        - audio         : the input audio file from the user
        - language      : the prefered language
        - chatbot       : the list of message between the user and the voice assistant
        - memory        : a list of variables to remember inbetween messages
        - whisper_model : the whisper model to use for STT
        - tts_models    : the TTS models to use
                          (dictionary with a TTS model for every available language)

    Output:
        the audio response, list of messages and memory
    """
    return talk(audio, language, whisper_model, tts_models, chatbot, memory)


# Gradio interface
with gr.Blocks(
    title="Voice Assistant",
    theme=gr.themes.Soft(primary_hue="emerald"),
    css="body {background-image: \
        url('https://dataroots.io/assets/logo/symbol-green.png');\
        background-size: 120px; background-repeat: round;}",
) as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center;"> Demo of the second project: \
        Voice assistant <h1>
        <h3 style="text-align: center;"> developed by Dataroots <h3>
        <h5 style="text-align: center;"> for more info: Sophie De Coppel \
        (sophie.decoppel@dataroots.io)  <h5>
        """
    )

    with gr.Column():
        # The prefered language
        language = gr.Radio(
            list(config.languages.keys()),
            value="Guess...",
            label="Which language do you speak?",
        )
    with gr.Row().style(equal_height=True):
        # The input and output audio
        audio_output = gr.Audio(
            value=audio_files[languages[language.value] or "en"], label="AI bot"
        )
        audio_input = gr.Audio(source="microphone", type="filepath", label="Human")
    with gr.Row().style(equal_height=True):
        talk_bttn = gr.Button("Submit")
        # flag_bttn_3 = gr.Button("Incorrect")
    with gr.Accordion("See conversation", open=False):
        # A written out version of the conversation
        chatbot = gr.Chatbot(
            value=[[None, start_messages[languages[language.value] or "en"]]],
            label="Conversation",
            show_label=False,
        )
        memory = gr.State()

        # Dynamicly change the welcome message when the user changes the language
        language.change(
            greet, [language, chatbot, audio_output], [language, audio_output, chatbot]
        )

    # Talk to the voice assistant
    talk_bttn.click(
        voice_assistant,
        inputs=[audio_input, language, chatbot, memory],
        outputs=[audio_output, language, chatbot, memory],
    )

    # Flag incorrect/inappropriate conversations
    callback = gr.CSVLogger()
    callback.setup([audio_input, language, audio_output, chatbot, memory], "flagged")
    # flag_bttn_3.click(
    #     lambda *args: callback.flag(args, flag_option="incorrect"),
    #     [audio_input, language],
    #     None,
    #     preprocess=False,
    # )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
