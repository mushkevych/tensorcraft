Tensor Craft is a collection of utilities and how-to for a typical AI/ML project.  
Here, the ImgProj stands for a Image Classification project with EfficientNet CNN Architecture

* [Builds the dataset](img_dataset_prep.ipynb) prepares the dataset
* [Trains the Model](img_classifier.ipynb) trains the model
* [computes CAM](img_grad_cam.ipynb) computes Gradient-weighted Class Activation Mapping to obtain a *heat map* visualization that highlights areas of an image that contribute to a neural network’s decision
* [Smoke Test Notebook](img_smoke_tester.ipynb) trains the model

### Intent
The intent of the ImgProj project is to illustrate end-to-end training of the CNN-based model

Tensorboard instructions
```bash
source ~/virtualenvs/tensor_craft/bin/activate
tensorboard --samples_per_plugin "images=100" --logdir ~/workspace/tensor_craft/tensorboard.run --bind_all --port 6006 --reuse_port True serve
# open browser at: echo http://localhost:6006
```
