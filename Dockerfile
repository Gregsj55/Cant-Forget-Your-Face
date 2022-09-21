FROM python:3.7
ADD . /code
WORKDIR /code
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=0
#CMD python3 apptest.py
ENTRYPOINT ["./gunicorn.sh"]