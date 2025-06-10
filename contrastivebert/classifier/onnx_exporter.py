from os import path

import onnxruntime as ort
import torch

from utils.compute_device import DEVICES
from utils.fs_utils import get_module_location
from contrastivebert.classifier.contrastivebert_classifier import ContrastiveSBERT
from contrastivebert.classifier.contrastivebert_configuration import ModelConf

ONNX_FILE_NAME = f'{ContrastiveSBERT.__name__}.onnx'
EVAL_METRICS_FILE_NAME = 'evaluation_metrics.json'
SYS_METRICS_FILE_NAME = 'system_metrics.json'
WEIGHTS_FILE_NAME = f'{ContrastiveSBERT.__name__}.weights'

cwd = get_module_location()


def export_to_onnx():
    device_name = 'cpu'
    model_conf = ModelConf()
    model = ContrastiveSBERT(model_conf)
    model.load_model_weights(path.join(cwd, WEIGHTS_FILE_NAME))
    model = model.to(device=DEVICES[device_name])

    # size=(1, model_conf.input_size) - here 1 stands for batch dimension
    dummy_input_ids = torch.randint(
        low=0, high=model.ml_components.tokenizer.vocab_size, size=(1, ModelConf.input_size), dtype=torch.long, device=DEVICES[device_name]
    )
    dummy_attention_mask = torch.randint(
        low=0, high=2, size=(1, ModelConf.input_size), dtype=torch.long, device=DEVICES[device_name]
    )

    dynamic_axes = {
        'input_ids': {0: 'batch_size'},       # Indicate dynamic batch size for input
        'attention_mask': {0: 'batch_size'},  # Indicate dynamic batch size for input
        'pooler_output': {0: 'batch_size'}    # Dynamic batch dimension for output
    }

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        f=path.join(cwd, ONNX_FILE_NAME),
        export_params=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['pooler_output'],
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
