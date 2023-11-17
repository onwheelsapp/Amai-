FROM python:3.11

# install dependencies
RUN apt-get update && apt-get install ffmpeg espeak rustc -y 

WORKDIR ./app
COPY pyproject.toml ./
RUN pip install "poetry==1.5.1"
RUN poetry config virtualenvs.create false
COPY src ./src
RUN poetry install

# run app
COPY docker/frontend .
ENTRYPOINT python gradio_app.py