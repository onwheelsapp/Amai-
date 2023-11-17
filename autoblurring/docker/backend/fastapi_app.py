from enum import Enum
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from src.auto_blurring import groundedSAM, maskRCNN

app = FastAPI()


class DetectionModel(str, Enum):
    maskRCNN = "maskRCNN"
    groundedSAM = "groundedSAM"


# load models
method1 = maskRCNN()
method2 = groundedSAM()


def anonymize_image(
    image_path: str,
    detection_model: DetectionModel = DetectionModel.maskRCNN,
    threshold1: float = 0.35,
    threshold2: float = 0.25,
) -> Image.Image:
    """Function wrapper to anonimize the image with the appropriate model and
    threshold(s)

    Input:
        - image_path        : path the to relevant image
        - detection_model   : the chosen anonimizatoin model (mask-RCNN or groundedSAM)
        - threshold1        : the threshold value for object recognition box
        - threshold2        : the threshold value for the text similarity
                              (only relevant for groundedSAM)

    Output:
        the anonimized image
    """

    # Perform object detection based on the selected model
    if detection_model == DetectionModel.maskRCNN:
        anonymized_img = method1(image_path, threshold=threshold1)

    elif detection_model == DetectionModel.groundedSAM:
        anonymized_img = method2(
            image_path,
            box_threshold=threshold1,
            text_threshold=threshold2,
        )

    else:
        raise ValueError("Invalid detection model selected")

    return anonymized_img


@app.post("/anonymize_image/")
async def process_image(
    file: UploadFile = File(...),
    model: DetectionModel = DetectionModel.maskRCNN,
    threshold1: float = 0.35,
    threshold2: float = 0.25,
) -> Any:
    """Function for post request to anonimize an image.

    Input:
        - file          : image file
        - model         : requested model for anonimization (maskRCNN or groundedSAM)
        - threshold1    : threhsold value for object recognition box
        - threshold2    : threshold value for text similarity
                          (only relevant for groundedSAM)

    Output:
        a file response with the anonimized image
    """

    # saving local copy of the image
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:  # type: ignore
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    if file.filename:
        # Perform image anonymization
        anonymized_image = anonymize_image(file.filename, model, threshold1, threshold2)
        anonymized_image.save("anonymized_image.png")

        # Return the anonymized image
        return FileResponse("anonymized_image.png", media_type="image/png")
    else:
        return {"message": "There was an error reading the file"}
