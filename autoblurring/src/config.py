import os
from dataclasses import dataclass


@dataclass
class model:
    # dataclass to store weights and config urls for a specific model
    weights: str
    config: str


@dataclass
class Urls(dict):
    # dataclass to store all relevant model information
    maskrcnn: model
    sam: model
    dino: model


# relevant download urls
urls = Urls(
    maskrcnn=model(
        weights="https://github.com/sambhav37/Mask-R-CNN/raw/master/mask-rcnn-coco/frozen_inference_graph.pb",  # noqa: E501,
        config="https://raw.githubusercontent.com/sambhav37/Mask-R-CNN/master/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt",  # noqa: E501
    ),
    sam=model(
        weights="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # noqa: E501
        config="",
    ),
    dino=model(
        weights="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",  # noqa: E501
        config="https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",  # noqa: E501
    ),
)


class Config:
    def __init__(self) -> None:
        self.models_path = self.get_models_path()
        os.makedirs(self.models_path, exist_ok=True)

        self.urls = urls

        # maskrcnn
        self.maskrcnn_dir = os.path.join(self.models_path, "mask-rcnn")
        os.makedirs(self.maskrcnn_dir, exist_ok=True)

        self.maskrcnn_weights_path = os.path.join(
            self.maskrcnn_dir, "frozen_inference_graph.pb"
        )
        self.maskrcnn_config_path = os.path.join(
            self.maskrcnn_dir, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        )

        # groundedsam
        self.groundedsam_dir = os.path.join(self.models_path, "groundedSAM")
        os.makedirs(self.groundedsam_dir, exist_ok=True)

        self.sam_weights_path = os.path.join(
            self.groundedsam_dir, "sam_vit_b_01ec64.pth"
        )
        self.dino_weights_path = os.path.join(
            self.groundedsam_dir, "groundingdino_swinb_cogcoor.pth"
        )
        self.dino_config_path = os.path.join(
            self.groundedsam_dir, "GroundingDINO_SwinB.cfg.py"
        )

    @staticmethod
    def get_models_path() -> str:
        if os.path.exists("../data/models/CVmodels"):
            return "../data/models/CVmodels"
        elif os.path.exists("../../data/models/CVmodels"):
            return "../../data/models/CVmodels"
        else:
            return "models"
