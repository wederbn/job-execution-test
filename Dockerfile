FROM python:3.7-slim
LABEL maintainer = "Benjamin Weder <weder@iaas.uni-stuttgart.de>"

COPY . /
RUN apt-get update && apt-get install -y gcc python3-dev
RUN pip install -r requirements.txt

CMD python app.py
