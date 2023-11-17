import os
import urllib.request

from src.config import Config

config = Config()

# maskrcnn models
os.makedirs(config.maskrcnn_dir, exist_ok=True)
if not os.path.exists(config.maskrcnn_weights_path):
    print("Download weights for maskrcnn")
    weightsPath, _ = urllib.request.urlretrieve(
        config.urls.maskrcnn.weights, filename=config.maskrcnn_weights_path
    )

if not os.path.exists(config.maskrcnn_config_path):
    print("Download configuration file for maskrcnn")
    configPath, _ = urllib.request.urlretrieve(
        config.urls.maskrcnn.config, filename=config.maskrcnn_config_path
    )

# groundedsam models
os.makedirs(config.groundedsam_dir, exist_ok=True)
if not os.path.exists(config.sam_weights_path):
    print("Download SAM")
    samweightsPath, _ = urllib.request.urlretrieve(
        config.urls.sam.weights, filename=config.sam_weights_path
    )

if not os.path.exists(config.dino_weights_path):
    print("Download weights for groundingDINO")
    dinoweightsPath, _ = urllib.request.urlretrieve(
        config.urls.dino.weights, filename=config.dino_weights_path
    )

if not os.path.exists(config.dino_config_path):
    print("Download configuration file for groundingDINO")
    dinoconfigPath, _ = urllib.request.urlretrieve(
        config.urls.dino.config, filename=config.dino_config_path
    )

print("All models downloaded succesfully!")
