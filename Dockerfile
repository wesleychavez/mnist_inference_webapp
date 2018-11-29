FROM gw000/keras:2.1.4-py3-tf-cpu

# Where to put files
ADD flask-app/ /tmp/flask-app
WORKDIR /tmp/flask-app

# Python libs
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y python3-pip
RUN pip3 --no-cache-dir install \
    Werkzeug \
    numpy \
    pillow \
    h5py \
    Flask==0.10.1

RUN pip3 --no-cache-dir install --upgrade keras

# Port number the container should expose
EXPOSE 5000

# Run the app
CMD ["python3", "./app.py"]
