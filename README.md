Tensor Craft is a collection of utilities and how-to for a typical AI/ML project.  

The `ImgProj` stands for an Image Classification project with EfficientNet CNN Architecture:
  * [img_dataset_prep.ipynb](imgproj/img_dataset_prep.ipynb) prepares the dataset
  * [img_classifier.ipynb](imgproj/img_classifier.ipynb) trains the model
  * [img_grad_cam.ipynb](imgproj/img_grad_cam.ipynb) computes Gradient-weighted Class Activation Mapping to obtain a *heat map* visualization that highlights areas of an image that contribute to a neural network’s decision
  * [img_smoke_tester.ipynb](imgproj/img_smoke_tester.ipynb) provides a notebook to run a smoke test

The `TxtProj` stands for a Text Classification project with SVM and Tfidf dictionary:
  * [txt_dataset_prep.ipynb](txtproj/txt_dataset_prep.ipynb) prepares the dataset
  * [txt_classifier.ipynb](txtproj/txt_classifier.ipynb) trains the model
  * [txt_smoke_tester.ipynb](txtproj/txt_smoke_tester.ipynb) provides a notebook to run a smoke test

The `MlpBertProj` stands for a Text Classification project of Bert Embeddings by the Multi-Level Perceptron:
  * [mlpbert_dataset_prep.ipynb](mlpbertproj/mlpbert_dataset_prep.ipynb) prepares the dataset
  * [mlpbert_classifier.ipynb](mlpbertproj/mlpbert_classifier.ipynb) trains the model
  * [mlpbert_smoke_tester.ipynb](mlpbertproj/mlpbert_smoke_tester.ipynb) provides a notebook to run a smoke test

The `XplainProj` stands for a Shapley Values Explanation project based of the LGBMClassifier model:
  * [xplain_dataset_prep.ipynb](xplainproj/xplain_dataset_prep.ipynb) prepares the dataset
  * [xplain_classifier.ipynb](xplainproj/xplain_classifier.ipynb) trains the model
  * [xplain_shap_values.ipynb](xplainproj/xplain_shap_values.ipynb) provides an illustration on how to use Shapley Permutation and Kernel Shapley
  * [xplain_smoke_tester.ipynb](xplainproj/xplain_smoke_tester.ipynb) provides a notebook to run a smoke test

The `LoraProj` stands for a LoRA fine-tuning of the BERT model:
  * [lora_dataset_prep.ipynb](loraproj/lora_dataset_prep.ipynb) prepares the dataset
  * [lora_classifier.ipynb](loraproj/lora_classifier.ipynb) fine-tunes the model

The `LlmAdapterProj` stands for an Adapter-based fine-tuning of the BERT model:
  * [llmadapter_dataset_prep.ipynb](llmadapterproj/llmadapter_dataset_prep.ipynb) prepares the dataset
  * [llmadapter_classifier.ipynb](llmadapterproj/llmadapter_classifier.ipynb) fine-tunes the model

`ContrastiveBert` illustrates contrastive learning of the BERT model, where "left" and "right" texts are embedded into the shared latent space:
  * [contrastivebert_classifier.ipynb](contrastivebert/contrastivebert_classifier.ipynb) tunes the model

`Handbooks` represent common operations for Numpy/Pandas:
  * [numpy_handbook.ipynb](handbooks/numpy_handbook.ipynb)
  * [pandas_handbook.ipynb](handbooks/pandas_handbook.ipynb)


Tensorboard instructions
```bash
source ~/virtualenvs/tensor_craft/bin/activate
tensorboard --samples_per_plugin "images=100" --logdir ~/workspace/tensor_craft/tensorboard.run --bind_all --port 6006 --reuse_port True serve

# open browser at: http://localhost:6006
```
