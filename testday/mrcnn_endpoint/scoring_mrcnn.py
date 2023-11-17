import json
import logging
import os
from typing import Dict

import cv2 as cv
import numpy as np

model: cv.dnn.Net


def init() -> None:
    """This function is called when the container is initialized/started, typically
    after create/update of the deployment.

    The logic is written here to perform init operations like caching the model in
    memory
    """
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    weights_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR") or ".", "mask-rcnn", "frozen_inference_graph.pb"
    )
    config_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR") or ".",
        "mask-rcnn",
        "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt",
    )
    model = cv.dnn.readNetFromTensorflow(weights_path, config_path)
    logging.info("Init complete")


def run(raw_data: str) -> Dict:
    """This function is called for every invocation of the endpoint to perform the
    actual prediction. It anonymizes the cars and people in the image.

    Input:
        raw data : a dictonary in string format containing
                    the text prompt, image and threshold for the model

    Output:
        a dictionary in string format with the image and an optional error message
    """

    logging.info("maskrcnn: request received")
    try:
        data = json.loads(raw_data)
    except Exception as e:
        return {"image": None, "message": e}

    if (
        "text_prompt" not in data.keys()
        or "image" not in data.keys()
        or "treshold" not in data.keys()
    ):
        logging.info("maskrcnn: request failed")
        return {
            "image": None,
            "message": "Request should contain a text prompt, image and treshold.",
        }

    # get labels
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
    ids_interest = [LABELS.index(lab.strip()) for lab in data["text_prompt"].split(".")]
    logging.info("maskrcnn: labels loaded")

    # prep image
    image = np.array(data["image"], dtype=np.uint8)
    (H, W) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, swapRB=False, crop=False)
    logging.info("maskrcnn: image loaded")

    # Get predictions
    model.setInput(blob)  # noqa: F821
    (boxes, masks) = model.forward(  # noqa: F821
        ["detection_out_final", "detection_masks"]
    )
    logging.info("maskrcnn: masks and boxes received")

    # get only relevant detections for relevant labels with a decent confidence
    idx = set(np.where(boxes[0, 0, :, 2] > data["treshold"])[0]).intersection(
        set(np.where(boxes[0, 0, :, 1] == ids_interest[0])[0])
    )
    if len(ids_interest) > 1:
        for id in ids_interest[1:]:
            idx = idx.union(set(np.where(boxes[0, 0, :, 1] == id)[0]))

    # loop over the number of detected objects
    for j in list(idx):
        classID = int(boxes[0, 0, j, 1])

        box = boxes[0, 0, j, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        boxW = endX - startX
        boxH = endY - startY

        if boxW > 0 and boxH > 0:
            mask = masks[j, classID]
            mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_CUBIC)
            mask = mask > 0.1

            # Blur the relevant parts of the image
            image[startY:endY, startX:endX][mask] = cv.GaussianBlur(
                image[startY:endY, startX:endX],
                (boxW // 4 * 2 + 1, boxH // 4 * 2 + 1),
                0,
            )[mask]

    logging.info("maskrcnn: request processed")
    return {"image": image.tolist(), "message": "Successfully anonymized"}
