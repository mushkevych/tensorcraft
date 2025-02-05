from os import path

import onnxruntime as ort
import torch

from fs_utils import get_module_location
from mlpbertproj.classifier.mlpbert_classifier import MlpBertModel
from mlpbertproj.classifier.mlpbert_configuration import ModelConf

ONNX_FILE_NAME = f'{MlpBertModel.__name__}.onnx'
EVAL_METRICS_FILE_NAME = 'evaluation_metrics.json'
SYS_METRICS_FILE_NAME = 'system_metrics.json'
WEIGHTS_FILE_NAME = f'{MlpBertModel.__name__}.weights'

cwd = get_module_location()


def export_to_onnx():
    model_conf = ModelConf()
    model = MlpBertModel(model_conf)
    model.load_model_weights(path.join(cwd, WEIGHTS_FILE_NAME))

    dummy_input = torch.randn(size=(1, model_conf.input_size))  # 1 stands for batch dimension
    dynamic_axes = {
        'np_text_embeddings': {0: 'batch_size'},      # Indicate dynamic batch size for input
        'np_logits': {0: 'batch_size'}                  # Dynamic batch dimension for output
    }

    torch.onnx.export(
        model,
        dummy_input,
        f=path.join(cwd, ONNX_FILE_NAME),
        export_params=True,
        input_names=['np_text_embeddings'],
        output_names=['np_logits'],
        dynamic_axes=dynamic_axes
    )


def print_details():
    session = ort.InferenceSession(path.join(cwd, ONNX_FILE_NAME))

    # Print the model's input information
    print('Inputs:')
    for input_tensor in session.get_inputs():
        print(input_tensor.name, input_tensor.shape)

    print('Outputs:')
    for output_tensor in session.get_outputs():
        print(output_tensor.name, output_tensor.shape)

    model_meta = session.get_modelmeta()
    print('Model Metadata:')
    print('Producer Name:', model_meta.producer_name)
    print('Graph Name:', model_meta.graph_name)
    print('Domain:', model_meta.domain)
    print('Description:', model_meta.description)
    print('Version:', model_meta.version)
    print('Custom Metadata Map:', model_meta.custom_metadata_map)

    session_options = session.get_session_options()
    print('\nSession Options:')
    print('Execution Mode:', session_options.execution_mode)
    print('Graph Optimization Level:', session_options.graph_optimization_level)


if __name__ == '__main__':
    export_to_onnx()
    print_details()
