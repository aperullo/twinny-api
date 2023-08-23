FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubi8

RUN dnf install -y python3.11 python3.11-pip

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY api.py model.py configuration.py /

RUN mkdir /models

CMD ["python3", "api.py"]

