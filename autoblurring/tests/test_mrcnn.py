import os
from importlib import resources

from PIL import Image, ImageChops
from src.auto_blurring import maskRCNN


class Test_maskrcnn:
    """Test functionalities of the maskrcnn class."""

    def test_load_models(self) -> None:
        """Test if the model files are downloaded."""
        m = maskRCNN()
        assert os.path.exists(m.config.maskrcnn_weights_path)
        assert os.path.exists(m.config.maskrcnn_config_path)

    def test_call_change(self) -> None:
        """Test if image is changed after call."""
        m = maskRCNN()
        with resources.path("tests.files", "normal_img.jpg") as img_path:
            img_before = Image.open(img_path)
            img_after = m(str(img_path))
            diff = ImageChops.difference(img_before, img_after)
            assert diff.getbbox()

    def test_call_same(self) -> None:
        """Test if black image is not changed after call."""
        m = maskRCNN()
        with resources.path("tests.files", "black_img.jpg") as img_path:
            img_before = Image.open(img_path)
            img_after = m(str(img_path))
            diff = ImageChops.difference(img_before, img_after)
            assert not diff.getbbox()
