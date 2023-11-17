import os
from importlib import resources

from PIL import Image, ImageChops
from src.auto_blurring import groundedSAM


class Test_groundedsam:
    """Test functionalities of the groundedsam class."""

    def test_load_models(self) -> None:
        """Test if the model files are downloaded."""
        gs = groundedSAM()
        assert os.path.exists(gs.config.sam_weights_path)
        assert os.path.exists(gs.config.dino_config_path)
        assert os.path.exists(gs.config.dino_weights_path)

    def test_call_change(self) -> None:
        """Test if image is changed after call."""
        gs = groundedSAM()
        with resources.path("tests.files", "normal_img.jpg") as img_path:
            img_before = Image.open(img_path)
            img_after = gs(str(img_path))
            diff = ImageChops.difference(img_before, img_after)
            assert diff.getbbox()

    def test_call_same(self) -> None:
        """Test if black image is not changed after call."""
        gs = groundedSAM()
        with resources.path("tests.files", "black_img.jpg") as img_path:
            img_before = Image.open(img_path)
            img_after = gs(str(img_path))
            diff = ImageChops.difference(img_before, img_after)
            assert not diff.getbbox()
