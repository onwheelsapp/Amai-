import os
from typing import Dict


class Config:
    def __init__(self) -> None:
        self.models_path = self.get_models_path()
        os.makedirs(self.models_path, exist_ok=True)
        self.audio_dir = self.get_audio_path()
        os.makedirs(self.audio_dir, exist_ok=True)

        # self.openai_api_type = "azure"
        # self.openai_api_base = "https://openai-onwheels.openai.azure.com/"
        # self.openai_api_version = "2023-03-15-preview"
        self.openai_api_type = "openai"
        self.openai_api_base = "https://api.openai.com/v1/chat/completions"
        self.openai_api_version = "gpt-3.5-turbo"

        self.languages = {
            "English": "en",
            "Nederlands": "nl",
            "Francais": "fr",
            "Deutsch": "de",
            "Guess...": None,
        }
        self.lang = [val for val in self.languages.values() if val is not None]

        self.chatbot_configs_dir = self.get_src_path()
        self.railfile_path = os.path.join(
            self.chatbot_configs_dir, "questionnaire.rail"
        )
        self.fail_messages_path = os.path.join(
            self.chatbot_configs_dir, "fail_messages.txt"
        )
        self.start_messages_path = os.path.join(
            self.chatbot_configs_dir, "start_messages.txt"
        )
        self.questions_path = {
            lang: os.path.join(self.chatbot_configs_dir, f"questions_{lang}.txt")
            for lang in self.lang
        }

        self.tts_model_paths = self.get_tts_model_paths()
        self.whisper_dir = os.path.join(self.models_path, "whisper")

    @staticmethod
    def get_src_path() -> str:
        if os.path.exists("src/chatbot_configs"):
            return "src/chatbot_configs"
        elif os.path.exists("../src/chatbot_configs"):
            return "../src/chatbot_configs"
        else:
            return "."

    @staticmethod
    def get_models_path() -> str:
        if os.path.exists("../data/models/NLPmodels"):
            return "../data/models/NLPmodels"
        elif os.path.exists("../../data/models/NLPmodels"):
            return "../../data/models/NLPmodels"
        else:
            return "models"

    @staticmethod
    def get_audio_path() -> str:
        if os.path.exists("../data/audio"):
            return "../data/audio"
        elif os.path.exists("../../data/audio"):
            return "../../data/audio"
        else:
            return "audio"

    def get_tts_model_paths(self) -> Dict[str, Dict[str, None | str]]:
        tts_model_paths = {
            "en": {
                "model_name": "tts_models/en/ljspeech/vits",
                "model_path": f"{self.models_path}/tts/tts_models--en--ljspeech--vits/model_file.pth",  # noqa: E501
                "config_path": f"{self.models_path}/tts/tts_models--en--ljspeech--vits/config.json",  # noqa: E501
                "vocoder_path": None,
                "vocoder_config_path": None,
                "tts_speakers_file": None,
                "tts_languages_file": None,
            },
            "nl": {
                "model_name": "tts_models/nl/css10/vits",
                "model_path": f"{self.models_path}/tts/tts_models--nl--css10--vits/model_file.pth.tar",  # noqa: E501
                "config_path": f"{self.models_path}/tts/tts_models--nl--css10--vits/config.json",  # noqa: E501
                "vocoder_path": None,
                "vocoder_config_path": None,
                "tts_speakers_file": f"{self.models_path}/tts/tts_models--nl--css10--vits/speaker_ids.json",  # noqa: E501
                "tts_languages_file": f"{self.models_path}/tts/tts_models--nl--css10--vits/language_ids.json",  # noqa: E501
            },
            "fr": {
                "model_name": "tts_models/fr/css10/vits",
                "model_path": f"{self.models_path}/tts/tts_models--fr--css10--vits/model_file.pth.tar",  # noqa: E501
                "config_path": f"{self.models_path}/tts/tts_models--fr--css10--vits/config.json",  # noqa: E501
                "vocoder_path": None,
                "vocoder_config_path": None,
                "tts_speakers_file": f"{self.models_path}/tts/tts_models--fr--css10--vits/speaker_ids.json",  # noqa: E501
                "tts_languages_file": f"{self.models_path}/tts/tts_models--fr--css10--vits/language_ids.json",  # noqa: E501
            },
            "de": {
                "model_name": "tts_models/de/thorsten/vits",
                "model_path": f"{self.models_path}/tts/tts_models--de--thorsten--vits/model_file.pth",  # noqa: E501
                "config_path": f"{self.models_path}/tts/tts_models--de--thorsten--vits/config.json",  # noqa: E501
                "vocoder_path": None,
                "vocoder_config_path": None,
                "tts_speakers_file": None,
                "tts_languages_file": None,
            },
        }
        return tts_model_paths
