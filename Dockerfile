FROM circleci/python:3.6.1

RUN set -x \ && apk add --no-cache build-base \ && apk add --no-cache libexecinfo-dev
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN pip install -U pip