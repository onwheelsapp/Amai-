import os

import torch
import whisper
from src.config import Config
from TTS.api import TTS
from TTS.utils.manage import ModelManager

config = Config()

lang_list = config.lang
os.makedirs(config.whisper_dir, exist_ok=True)
whisper_model = whisper.load_model("medium", download_root=config.whisper_dir)
gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print("All models downloaded succesfully!")
