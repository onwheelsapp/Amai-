FROM python:3.10

# install dependencies
WORKDIR ./app
COPY pyproject.toml .
COPY src ./src
RUN pip install "poetry==1.5.1"
RUN poetry config virtualenvs.create false
RUN poetry install

# run app
COPY docker/frontend .
ENTRYPOINT python gradio_app.py