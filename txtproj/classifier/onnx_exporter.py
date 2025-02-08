from os import path

import onnxmltools
import onnxruntime as ort
from skl2onnx.common.data_types import FloatTensorType

from fs_utils import get_module_location
from txtproj.classifier.txt_classifier import LrTxtClassifier, SvmTxtClassifier, LgbmTxtClassifier
from txtproj.classifier.txt_configuration import ModelConf

TMPL_ONNX_FILE_NAME = '{0}.onnx'
TMPL_EVAL_METRICS_FILE_NAME = 'evaluation_metrics.{0}.json'
TMPL_SYS_METRICS_FILE_NAME = 'system_metrics.{0}.json'
TMPL_WEIGHTS_FILE_NAME = '{0}.weights'

cwd = get_module_location()


def export_to_onnx(outlier_class: type[LrTxtClassifier | SvmTxtClassifier]):  # LgbmTextClassifier
    model_conf = ModelConf()
    model = outlier_class(model_conf)
    model.load_model_weights(path.join(cwd, TMPL_WEIGHTS_FILE_NAME.format(outlier_class.__name__)))

    initial_type: list = [('text_tfidf', FloatTensorType([None, model_conf.input_size]))]
    onnx_model = onnxmltools.convert_sklearn(model._model, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, TMPL_ONNX_FILE_NAME.format(outlier_class.__name__))


def print_details(outlier_class: type[LrTxtClassifier | SvmTxtClassifier | LgbmTxtClassifier]):
    session = ort.InferenceSession(TMPL_ONNX_FILE_NAME.format(outlier_class.__name__))
    print('-' * 10)
    print(f'Model classname: {outlier_class.__name__}')

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
    for outlier_class in [LrTxtClassifier, SvmTxtClassifier]:  #, LgbmTxtClassifier
        export_to_onnx(outlier_class)
        print_details(outlier_class)
