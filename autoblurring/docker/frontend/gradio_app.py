# general imports
import gradio as gr
from PIL import Image

# helper files
from src.auto_blurring import groundedSAM, maskRCNN

# link to the example images
example_images = [
    "examples/entrance/0a01ffc9b28020381ca7396a57bcca3f_image_entrance_1.jpg",
    "examples/parking/0a73189e6a9160bfb31434117b793fba_image_parking_1.jpg",
    "examples/parking/0b730c20b8a40e2e0bac3213b6cc3088_image_parking_1.jpg",
]

# load models
method1 = maskRCNN()
method2 = groundedSAM()


def process_image(img: str, model_choice: str = "maskRCNN") -> Image.Image:
    """Wrapper function to process the image with the chosen anonimization model.

    Input:
        - img           : the image
        - model_choice  : the chosen model (either groundedSAM or maskRCNN)
    """
    if model_choice == "groundedSAM":
        output_img = method2(img)
    elif model_choice == "maskRCNN":
        output_img = method1(img)
    else:
        output_img = Image.open(img)
    return output_img


# Gradio interface of the demo
with gr.Blocks(
    title="Auto-blurring",
    theme=gr.themes.Soft(primary_hue="emerald"),
    css="body {background-image: \
        url('https://dataroots.io/assets/logo/symbol-green.png'); \
        background-size: 120px; background-repeat: round;}",
) as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center;"> Demo of the first project: Auto-blurring <h1>
        <h3 style="text-align: center;"> developed by Dataroots <h3>
        <h5 style="text-align: center;"> for more info: Sophie De Coppel \
        (sophie.decoppel@dataroots.io)  <h5>
        """
    )

    # Radio button to choose the anonimization model
    model_choice = gr.Radio(
        ["maskRCNN", "groundedSAM"], value="maskRCNN", label="Model"
    )

    # Input and output block for the images
    with gr.Column():
        input_img = gr.Image(type="filepath", label="Image")
        go_bttn = gr.Button("Anonymize")
        output_img = gr.Image(label="Anonimized image")

        go_bttn.click(
            process_image, inputs=[input_img, model_choice], outputs=output_img
        )

    # Example images
    gr.Examples(
        examples=example_images,
        inputs=input_img,
        outputs=output_img,
        fn=process_image,
        cache_examples=True,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
