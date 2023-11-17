import os

from src.config import Config
from TTS.api import TTS
from TTS.utils.manage import ModelManager


class Test_config:
    """Test cases for configuration file."""

    def test_model_path(self) -> None:
        """Check if model paths exists."""
        config = Config()
        assert os.path.exists(config.models_path)
        assert os.path.exists(config.audio_dir)
        assert os.path.exists(config.chatbot_configs_dir)

    def test_existance_ttsmodels(self) -> None:
        """Check if the required tts models exist."""
        config = Config()
        manager = ModelManager(
            models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False
        )
        available_models = manager.list_tts_models()
        for model in config.tts_model_paths.values():
            assert model["model_name"] in available_models
