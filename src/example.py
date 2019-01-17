from fastai import *
from fastai.vision import *

path = untar_data(URLs.MNIST_SAMPLE)

print('Using path %s' % path)
data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)
img, label = data.train_ds[0]

learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)

accuracy(*learn.get_preds())
