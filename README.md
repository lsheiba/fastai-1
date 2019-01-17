# PyTORCH

This is the base PyTORCH project.

Contains:

- Jupyter for interactive development
- Examples of training tasks

Repository pytorch-examples contains example of spatial transformation using
MNIST dataset. If you want to see the details about transformation, please look at
jupyter notebook **spatial_transformer_tutorial.ipynb**
in the **src** directory using **Jupyter** after project installation.

Steps to run spatial transformer as a separate task or on GPU:

- Install project with default source repo and config
- Go to tab "Tasks"
- Choose "standalone" -> "worker"
- (Optional) Adjust resource requests to use GPU: set GPU to **1**
- Click "Save and execute"
- Then, watch the logs in tab "Logs": expand row named "standalone:X"
- Wait unitl job is succeeded
- Open "Jupyter" tab and look into **training** dir: there should be 2 **.png** files
containing a set of test images and a set of corresponding processed images. You can download them
and see the result of transformation.