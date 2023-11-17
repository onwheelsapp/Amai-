import os

import requests
from src.config import Config


class Test_config:
    """Test cases for configuration file."""

    def test_model_path(self) -> None:
        """Check if model paths exists."""
        config = Config()
        assert os.path.exists(config.models_path)
        assert os.path.exists(config.maskrcnn_dir)
        assert os.path.exists(config.groundedsam_dir)

    def test_download_urls(self) -> None:
        """Check if urls are valid."""
        config = Config()
        for model in config.urls:
            if model.weights:
                response = requests.get(model.weights)
                assert response.status_code == 200
            if model.config:
                response = requests.get(model.weights)
                assert response.status_code == 200
