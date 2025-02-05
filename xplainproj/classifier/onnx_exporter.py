from os import path

import onnxmltools
import onnxruntime as ort
from skl2onnx.common.data_types import FloatTensorType

from fs_utils import get_module_location
from xplainproj.classifier.xplain_classifier import XplainClassifier
from xplainproj.classifier.xplain_configuration import ModelConf

ONNX_FILE_NAME = f'{XplainClassifier.__name__}.onnx'
EVAL_METRICS_FILE_NAME = 'evaluation_metrics.json'
SYS_METRICS_FILE_NAME = 'system_metrics.json'
WEIGHTS_FILE_NAME = f'{XplainClassifier.__name__}.weights'

cwd = get_module_location()


def export_to_onnx():
    model_conf = ModelConf()
    model = XplainClassifier(model_conf)
    model.load_model_weights(path.join(cwd, WEIGHTS_FILE_NAME))

    initial_type: list = [
        ('np_features', FloatTensorType([None, model_conf.input_size]))
    ]
    onnx_model = onnxmltools.convert_lightgbm(model._model, initial_types=initial_type)

    # `convert_lightgbm` sets name of the output tensor to 'label' and does not allow to change it
    # code below changes the output shape from [1] to [None], where 'None' stands for 'batch' dimension
    for output_tensor in onnx_model.graph.output:
        if output_tensor.name == 'label':
            output_tensor.type.tensor_type.shape.dim[0].dim_param = 'N'

    onnxmltools.utils.save_model(onnx_model, path.join(cwd, ONNX_FILE_NAME))


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
