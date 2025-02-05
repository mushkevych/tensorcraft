Tensor Craft is a collection of utilities and how-to for a typical AI/ML project.  

`The ImgProj` stands for an Image Classification project with EfficientNet CNN Architecture:
  * [img_dataset_prep.ipynb](imgproj/img_dataset_prep.ipynb) prepares the dataset
  * [img_classifier.ipynb](imgproj/img_classifier.ipynb) trains the model
  * [img_grad_cam.ipynb](imgproj/img_grad_cam.ipynb) computes Gradient-weighted Class Activation Mapping to obtain a *heat map* visualization that highlights areas of an image that contribute to a neural network’s decision
  * [img_smoke_tester.ipynb](imgproj/img_smoke_tester.ipynb) provides a notebook to run a smoke test

`The TxtProj` stands for a Text Classification project with SVM and Tfidf dictionary:
  * [txt_dataset_prep.ipynb](txtproj/txt_dataset_prep.ipynb) prepares the dataset
  * [txt_classifier.ipynb](txtproj/txt_classifier.ipynb) trains the model
  * [txt_smoke_tester.ipynb](txtproj/txt_smoke_tester.ipynb) provides a notebook to run a smoke test

`The MlpBertProj` stands for a Text Classification project of Bert Embeddings by the Multi-Level Perceptron:
  * [mlpbert_dataset_prep.ipynb](mlpbertproj/mlpbert_dataset_prep.ipynb) prepares the dataset
  * [mlpbert_classifier.ipynb](mlpbertproj/mlpbert_classifier.ipynb) trains the model
  * [mlpbert_smoke_tester.ipynb](mlpbertproj/mlpbert_smoke_tester.ipynb) provides a notebook to run a smoke test

`The XplainProj` stands for a Shapley Values Explanation project based of the LGBMClassifier model:
  * [xplain_dataset_prep.ipynb](xplainproj/xplain_dataset_prep.ipynb) prepares the dataset
  * [xplain_classifier.ipynb](xplainproj/xplain_classifier.ipynb) trains the model
  * [xplain_shap_values.ipynb](xplainproj/xplain_shap_values.ipynb) provides an illustration on how to use Shapley Permutation and Kernel Shapley
  * [xplain_smoke_tester.ipynb](xplainproj/xplain_smoke_tester.ipynb) provides a notebook to run a smoke test

Tensorboard instructions
```bash
source ~/virtualenvs/tensor_craft/bin/activate
tensorboard --samples_per_plugin "images=100" --logdir ~/workspace/tensor_craft/tensorboard.run --bind_all --port 6006 --reuse_port True serve

# open browser at: http://localhost:6006
```
