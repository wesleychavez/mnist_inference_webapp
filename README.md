# MNIST CNN training and Flask inference app


## Flask inference app
Now available on the Docker Hub! 
```shell
docker pull weschavezforsure/mnist_inference
docker run -p 5000:5000 weschavezforsure/mnist_inference
```
or just
```shell
docker build -t mnist_inference .
docker run -p 5000:5000 mnist_inference
```
`flask_app` contains the web application for MNIST inference.  Upload any image, it will be converted to a 28x28 pixel grayscale image and the most likely digit and corresponding probability will be displayed.  The front-end of this app was cloned from https://github.com/mtobeiyf/keras-flask-deploy-webapp

## CNN training
```shell
python train.py
```
`config_train.py` has a couple model hyperparameters.
`train.py` builds a specific CNN, trains it, and saves the best model according to validation accuracy. The MNIST test set (10k images) is used as validation.  This model achieves about 99% accuracy on the MNIST test set.  This model takes about 45 minutes per epoch on a 2018 MacBook Pro and about 30 seconds per epoch on a AWS EC2 p2.xlarge

Requires:
- python 3
- tensorflow
- keras
