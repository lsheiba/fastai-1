from fastai import *

untar_data(MNIST_PATH)
data = image_data_from_folder(MNIST_PATH)
learn = create_cnn(data, tvm.resnet18, metrics=accuracy)
learn.fit(1)
