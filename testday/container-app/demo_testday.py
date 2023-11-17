# general imports
from typing import List, Tuple

import gradio as gr

# helper files
from autoblurring import process_image
from voiceassistance import audio_files, start_messages, voice_assistant

# in string path format to see the icons, otherwise icon is not correctly shown
example_images = [
    "examples/entrance/0a01ffc9b28020381ca7396a57bcca3f_image_entrance_1.jpg",
    "examples/parking/0a73189e6a9160bfb31434117b793fba_image_parking_1.jpg",
    "examples/parking/0b730c20b8a40e2e0bac3213b6cc3088_image_parking_1.jpg",
]

# available languages
languages = {
    "English": "en",
    "Nederlands": "nl",
    "Francais": "fr",
    "Deutsch": "de",
    "Guess...": "en",
}


def greet(
    language: str, chatbot: List[Tuple[str | None, str | None]], audio_output: str
) -> Tuple[str, str, List[Tuple[str | None, str | None]]]:
    """Function to dynamicly change the welcome message based on the language the user
    chooses.

    Input:
        - language      : the preferd language
        - chatbot       : a list of messages with the chatbot
        - audio_output  : the welcome message

    Output:
        - language      : the prefered language
        - audio_output  : the corresponding welcome message (audio)
        - chatbot       : the corresponding welcome message (text)
    """
    if len(chatbot) == 1:
        audio_output = audio_files[languages[language]]
        chatbot = [(None, start_messages[languages[language]])]
    return language, audio_output, chatbot


# interface for both projects
with gr.Blocks(
    title="Demo of the On Wheels project",
    theme=gr.themes.Soft(primary_hue="emerald"),
    css="body {background-image: \
        url('https://dataroots.io/assets/logo/symbol-green.png'); \
        background-size: 120px; background-repeat: round;}",
) as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center;"> Demo of the projects <h1>
        <h3 style="text-align: center;"> developed by Dataroots <h3>
        <h5 style="text-align: center;"> for more info: Sophie De Coppel
        (sophie.decoppel@dataroots.io)  <h5>
        """
    )

    # A tab with explanation about the project
    with gr.Tab("Explanation"):
        gr.Markdown(
            """
            ## Context and a short explanation about this demo  \
            This demo let's you experiment with **2 AI projects** \
            that we created. These are just proof-of-concepts, \
            so they are not final. \
            Therefore we would greatly appriciate your input!\n\n \
            You can explore them by clicking one of the tabs above \
            (Auto-blurring or Voice assistant).\n\n  \
            If you encounter any problems during the demo, \
            feel free to leave a message in the **feedback form**. \
            Positive feedback is, of course, also welcome!\n   \
            Thank you for your help!
            """
        )

    # Gradio interface for the first project
    with gr.Tab("Auto-blurring"):
        gr.Markdown(
            """
                The **first project**, called auto-blurring, \
                will try to protect the privacy of users by \
                automaticly anonymizing people \
                and vehicles in an image. You can give it an \
                image and see the magic happen for yourself.\n\n   \
                If you don't have an image, you can try out the \
                examples below by clicking on them.
                """
        )
        with gr.Row().style(equal_height=True):
            with gr.Column():
                input_img = gr.Image(type="filepath", label="Image")
                go_bttn = gr.Button("Anonymize")
                output_img = gr.Image(label="Anonimized_image")
                gr.Examples(
                    examples=example_images,
                    inputs=input_img,
                    outputs=output_img,
                    fn=lambda img: process_image(img),
                    cache_examples=True,
                )
            go_bttn.click(
                lambda img: process_image(img), inputs=input_img, outputs=output_img
            )

    # Gradio interface for the second project
    with gr.Tab("Voice assistant"):
        gr.Markdown(
            """
            The **second project** is a voice-assistant that helps \
            you add a location to the OnWheels app. \
            You can speak to it through your microphone and it will \
            respond back with audio. It understands dutch, french, \
            german and english.\n\n   \
            Listen to the output of the AI bot on the left and speak \
            to it on the right. Below, you can also \
            follow up your conversation in text.
            """
        )
        with gr.Column():
            language = gr.Radio(
                ["English", "Nederlands", "Francais", "Deutsch", "Guess..."],
                value="Guess...",
                label="Language",
                info="Which language do you speak?",
            )
        with gr.Row().style(equal_height=True):
            audio_output = gr.Audio(
                value=audio_files[languages[language.value]], label="AI bot output"
            )
            audio_input = gr.Audio(
                source="microphone", type="filepath", label="Human input"
            )
        with gr.Row().style(equal_height=True):
            talk_bttn = gr.Button("Submit")
        with gr.Accordion("See conversation", open=False):
            chatbot = gr.Chatbot(
                value=[[None, start_messages[languages[language.value]]]],
                label="Conversation",
                show_label=False,
            )
            memory = gr.State()
            language.change(
                greet,
                [language, chatbot, audio_output],
                [language, audio_output, chatbot],
            )

        talk_bttn.click(
            voice_assistant,
            inputs=[audio_input, language, chatbot, memory],
            outputs=[audio_output, chatbot, memory],
        )

    # feedback form
    with gr.Tab("Feedback"):
        gr.Markdown(
            """
        ### Your feedback matters! \
        Please provide any feedback you have in any language. \
        This can be about the performance of the AI models, the accessibility, \
        user-friendlyness or anything else you think of.
        """
        )
        stars = gr.Radio(
            label="How would you rate this AI project?", choices=[1, 2, 3, 4, 5]
        )
        feedback = gr.Textbox(label="Your feedback")
        feedback_bttn = gr.ClearButton([stars, feedback], value="Send feedback")
        callback_feedback = gr.CSVLogger()
        callback_feedback.setup([feedback, stars], "feedback")
        feedback_bttn.click(
            lambda *args: callback_feedback.flag(args),
            [feedback, stars],
            None,
            preprocess=False,
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
    )
