# fastai

This is the base **fastai** project.

Contains:

- Jupyter for interactive development
- Examples of training tasks

The fastai library simplifies training fast and accurate neural nets using modern best practices.
See the [fastai website](https://docs.fast.ai) to get started. The library is based on research
into deep learning best practices undertaken at [fast.ai](http://www.fast.ai), and includes
\"out of the box\" support for [`vision`](https://docs.fast.ai/vision.html#vision),
[`text`](https://docs.fast.ai/text.html#text), [`tabular`](https://docs.fast.ai/tabular.html#tabular),
and [`collab`](https://docs.fast.ai/collab.html#collab) (collaborative filtering) models. For
brief examples, see the [examples](https://github.com/fastai/fastai/tree/master/examples) folder;
detailed examples are provided in the full [documentation](https://docs.fast.ai/). For instance,
here's how to train an MNIST model using [resnet18](https://arxiv.org/abs/1512.03385) (from the
[vision example](https://github.com/fastai/fastai/blob/master/examples/vision.ipynb)):

```python
untar_data(MNIST_PATH)
data = image_data_from_folder(MNIST_PATH)
learn = create_cnn(data, tvm.resnet18, metrics=accuracy)
learn.fit(1)
```
