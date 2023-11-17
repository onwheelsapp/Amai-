import json
import os
from typing import List, TypedDict, cast

import cv2 as cv
import numpy as np
import requests

# from dotenv import load_dotenv

# load_dotenv()

# model endpoint information
url = "https://mrcnn-onwheels-end.westeurope.inference.ml.azure.com/score"
api_key = os.getenv("MRCNN_END_KEY") or ""
headers = {
    "Content-Type": "application/json",
    "Authorization": ("Bearer " + api_key),
    "azureml-model-deployment": "mrcnn-1",
}


class AnonymizeResponseDict(TypedDict):
    image: List


class ResponseDict(TypedDict):
    text: AnonymizeResponseDict


def process_image(
    filepath: str,
    text_prompt: str = "person . car . motorcycle . bus . truck",
    box_treshold: float = 0.35,
) -> np.ndarray:
    """
    Processes an image through the whole pipeline:
    - maskRCNN to segmentate the object
    - gaussian blur to anonimize the object

    Input:
        - filepath              : the filepath to the image
        - text_prompt           : the text input for the model
        - box_treshold          : the treshold of probability that defines
                                  the size of the bounding box around the object

    Output:
        The image where the relvant objects are blurred (if present).
    """
    image = cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2RGB)
    print("image loaded")
    data = {
        "text_prompt": text_prompt,
        "image": image.tolist(),
        "treshold": box_treshold,
    }
    print("posting request to mrcnn endpoint")
    r = requests.post(url, data=json.dumps(data), headers=headers)
    print("received response from mrcnn endpoint")
    response_dict = cast(ResponseDict, r.json())
    # retrieve the response from the model endpoint
    try:
        resp = response_dict["text"]
        if resp["image"] is not None:
            return np.array(resp["image"])
        else:
            print("Error: something went wrong in the maskrcnn endpoint")
    except Exception as e:
        print("Error: something went wrong in the maskrcnn endpoint")
        print(e)
    return image
