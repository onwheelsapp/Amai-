FROM python:3.10

# install dependencies
WORKDIR ./app
COPY pyproject.toml .
COPY src ./src
RUN pip install "poetry==1.5.1"
RUN poetry config virtualenvs.create false
RUN poetry install

# run app
COPY docker/backend .
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"]