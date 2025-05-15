FROM python:3.9

RUN apt-get -y update && \
    apt-get install -y ffmpeg

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip list

COPY . /code

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8048", "--reload"]