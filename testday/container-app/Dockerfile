FROM python:3.8

# install dependencies
RUN apt-get update && apt-get install espeak ffmpeg libsm6 libxext6  -y
COPY ./requirements.txt /tmp/
RUN pip install -U pip && pip install --no-cache-dir -r /tmp/requirements.txt

# get all necessary files
COPY . ./app

# run app
WORKDIR app
EXPOSE 8080
CMD ["python" , "demo_testday.py"]

#docker buildx build --platform linux/amd64 -t testday-app . --output type=docker




