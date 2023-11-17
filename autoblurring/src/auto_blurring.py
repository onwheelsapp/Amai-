# basic packages
import os
import urllib.request
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch

# Grounding DINO
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, load_model, predict
from PIL import Image

# Segment Anything
from segment_anything import (
    SamPredictor,
    build_sam,
    build_sam_vit_b,
    build_sam_vit_l,
    predictor,
)

# Configurations
from src.config import Config


class maskRCNN:
    """Loading and calling functionality of the maskRCNN model."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.model = self.load_model()

    def load_model(
        self, weightsPath: str | None = None, configPath: str | None = None
    ) -> cv.dnn.Net:
        """Simple loading function for the maskRCNN model.

        Input:
            - weightsPath : relative path to the model weights
            - configPath  : relative path to the model configuration file

        Output:
            the maskRCNN model
        """
        weightsPath = weightsPath or self.config.maskrcnn_weights_path
        configPath = configPath or self.config.maskrcnn_config_path

        os.makedirs(self.config.maskrcnn_dir, exist_ok=True)
        if not os.path.exists(weightsPath):
            print("Download weights for maskrcnn")
            weightsPath, _ = urllib.request.urlretrieve(
                self.config.urls.maskrcnn.weights, filename=weightsPath
            )

        if not os.path.exists(configPath):
            print("Download configuration file for maskrcnn")
            configPath, _ = urllib.request.urlretrieve(
                self.config.urls.maskrcnn.config, filename=configPath
            )

        print("Start loading maskrcnn model...")
        maskrcnn = cv.dnn.readNetFromTensorflow(weightsPath, configPath)
        print("maskrcnn loaded")
        return maskrcnn

    def __call__(
        self,
        filepath: str | os.PathLike,
        text_prompt: str = "person . car . motorcycle . bus . truck",
        threshold: float = 0.3,
    ) -> Image.Image:
        """Anonimize an image with the MaskRCNN model.

        Input:
            - filepath      : the relative path the image
            - maskRCNN      : the maskRCNN model
            - text_prompt   : the insructive text prompt given to the model
                            (labels splitted by a dot ".")
            - threshold     : the threshold value for the object recognition box

        Output:
            the anonimized image
        """

        # identify the objects of interest
        LABELS = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "street sign",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "hat",
            "backpack",
            "umbrella",
            "shoe",
            "eye glasses",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "plate",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "mirror",
            "dining table",
            "window",
            "desk",
            "toilet",
            "door",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "blender",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        ids_interest = [
            LABELS.index(lab.strip())
            for lab in text_prompt.split(".")
            if lab.strip() in LABELS
        ]

        # load the image
        image = cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2RGB)
        (H, W) = image.shape[:2]
        print("loaded image")

        blob = cv.dnn.blobFromImage(image, swapRB=False, crop=False)
        self.model.setInput(blob)
        (boxes, masks) = self.model.forward(["detection_out_final", "detection_masks"])
        print("predicted boxes and masks")

        # get only relevant detections for relevant labels with a decent confidence
        idx = set(np.where(boxes[0, 0, :, 2] > threshold)[0]).intersection(
            set(np.where(boxes[0, 0, :, 1] == ids_interest[0])[0])
        )
        if len(ids_interest) > 1:
            for id in ids_interest[1:]:
                idx = idx.union(set(np.where(boxes[0, 0, :, 1] == id)[0]))

        # loop over the number of detected objects
        for j in list(idx):
            classID = int(boxes[0, 0, j, 1])

            # determine array of the object
            box = boxes[0, 0, j, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            mask = masks[j, classID]
            mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_CUBIC)
            mask = mask > 0.1

            # blur the object
            image[startY:endY, startX:endX][mask] = cv.GaussianBlur(
                image[startY:endY, startX:endX],
                (boxW // 4 * 2 + 1, boxH // 4 * 2 + 1),
                0,
            )[mask]

        if not len(list(idx)):
            print("nothing detected")

        return Image.fromarray(image)


class groundedSAM:
    """Loading and calling functionality for the groundedSAM model."""

    def __init__(self, config: Config | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or Config()
        self.sam_model, self.dino_model = self.load_models()

    def load_models(
        self,
        samweightsPath: str | None = None,
        dinoweightsPath: str | None = None,
        dinoconfigPath: str | None = None,
    ) -> Tuple[predictor.SamPredictor, GroundingDINO]:
        """Simple loading function for the groundedSAM model.

        Input:
            - samweightsPath : the relative path to the model weights of SAM
            - dinoweightsPath: the relative path to the mdoel weights of grounding DINO
            - dinoconfigPath : the relative path to the config file of grounding DINO

        Output:
            a tuple containing the SAM model and grounding DINO model
        """

        samweightsPath = samweightsPath or self.config.sam_weights_path
        dinoweightsPath = dinoweightsPath or self.config.dino_weights_path
        dinoconfigPath = dinoconfigPath or self.config.dino_config_path

        os.makedirs(self.config.groundedsam_dir, exist_ok=True)
        if samweightsPath is None or not os.path.exists(samweightsPath):
            print("Download SAM")
            samweightsPath, _ = urllib.request.urlretrieve(
                self.config.urls.sam.weights, filename=samweightsPath
            )

        if dinoweightsPath is None or not os.path.exists(dinoweightsPath):
            print("Download weights for groundingDINO")
            dinoweightsPath, _ = urllib.request.urlretrieve(
                self.config.urls.dino.weights, filename=dinoweightsPath
            )

        if dinoconfigPath is None or not os.path.exists(dinoconfigPath):
            print("Download configuration file for groundingDINO")
            dinoconfigPath, _ = urllib.request.urlretrieve(
                self.config.urls.dino.config, filename=dinoconfigPath
            )

        print("Start loading SAM model...")
        if "vit_b" in samweightsPath:
            sam_predictor = SamPredictor(
                build_sam_vit_b(checkpoint=samweightsPath).to(self.device)
            )
        elif "vit_l" in samweightsPath:
            sam_predictor = SamPredictor(
                build_sam_vit_l(checkpoint=samweightsPath).to(self.device)
            )
        else:
            sam_predictor = SamPredictor(
                build_sam(checkpoint=samweightsPath).to(self.device)
            )
        print("SAM model loaded")

        print("Start loading groundingDINO model...")
        groundingdino_model = load_model(dinoconfigPath, dinoweightsPath, self.device)
        print("groundingDINO model loaded")

        return (sam_predictor, groundingdino_model)

    @staticmethod
    def blur_mask(
        image: np.ndarray, masks: List[np.ndarray], boxes: torch.Tensor
    ) -> np.ndarray:
        """Blurs an image according to the given masks from grounedSAM with a Gaussian
        blur. The gaussian blur is scaled according to the size of the bounding box.

        Input:
            - image : the image to blur (in numpy format)
            - masks : a list of masks
            (if the device is a gpu, the masks are transferred to the cpu)
            - boxes : a list of the corresponding bounding boxes
            (if the device is a gpu, the masks are transferred to the cpu)
            - device : the device that was used to run the models

        Output:
            The blurred image.
        """
        temp_image = image.copy()
        blurred_image = image.copy()

        for mask, box in zip(masks, boxes):
            (x1, y1, x2, y2) = box.type(torch.int64).cpu().numpy()
            temp_image[y1:y2, x1:x2, :] = cv.GaussianBlur(
                temp_image[y1:y2, x1:x2, :],
                ((x2 - x1) // 4 * 2 + 1, (y2 - y1) // 4 * 2 + 1),
                0,
            )
            blurred_image[mask] = temp_image[mask]

        return blurred_image

    def __call__(
        self,
        filepath: str | os.PathLike,
        text_prompt: str = "person . car . motorcycle . bus . truck",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Image.Image:
        """Anonimize an image with groundedSAM.

        Input:
            - filepath              : the relative path to the image file
            - sam_predictor         : the SAM model
            - groundingdino_model   : the groundingDINO model
            - text_prompt           : the text prompt given to the groundingDINO model
                                    (different lables should be seperated by a dot ".")
            - box_threshold         : threshold value for the object recognition box
            - text_threshold        : threshold value for the text similarity
        """

        # load image
        image_before, image = load_image(filepath)
        print("loaded image")

        # predict boxes with Grounding DINO
        boxes, _, _ = predict(
            model=self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        print("predicted boxes")
        if boxes.nelement():

            # predict masks with Segment Anything model
            self.sam_model.set_image(image_before)
            H, W, _ = image_before.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            transformed_boxes = self.sam_model.transform.apply_boxes_torch(
                boxes_xyxy, image_before.shape[:2]
            ).to(self.device)
            masks, _, _ = self.sam_model.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks_input = [masks[i][0] for i in range(len(masks))]
            if self.device != "cpu":
                masks_input = [mask.cpu().numpy() for mask in masks_input]
            print("predicted masks")

            # Blurring with Gaussian blur
            return Image.fromarray(
                self.blur_mask(image_before, masks_input, boxes_xyxy)
            )

        # No relevant objects detected
        else:
            print("Nothing detected")
            return Image.fromarray(image_before)
