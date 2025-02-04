import onnxruntime as ort
import torch
import torch.onnx

from imgproj.classifier.img_classifier import ImgClassifier
from imgproj.classifier.img_configuration import ModelConf

TMPL_ONNX_FILE_NAME = '{0}.{1}px.onnx'
TMPL_EVAL_METRICS_FILE_NAME = 'evaluation_metrics.{0}.{1}px.json'
TMPL_SYS_METRICS_FILE_NAME = 'system_metrics.{0}.{1}px.json'
TMPL_WEIGHTS_FILE_NAME = '{0}.{1}px.weights'

COMPUTE_DEVICE_CPU = torch.device('cpu')

def export_to_onnx(outlier_class: type[ImgClassifier]):
    model_conf = ModelConf()
    model = outlier_class(model_conf)
    model.load_model_weights(TMPL_WEIGHTS_FILE_NAME.format(outlier_class.__name__, model_conf.image_size[0]))

    # size=(1, 1, model_conf.image_size, model_conf.image_size) stands for (1 batch, 1 color channel, image_height, image_width)
    # * 255 stands for EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range
    dummy_input = torch.rand(size=(1, 1, model_conf.image_size[0], model_conf.image_size[1]), device=COMPUTE_DEVICE_CPU) * 255
    dynamic_axes = {
        'img_grey': {0: 'batch_size'},    # Indicate dynamic batch size for input
        'logits': {0: 'batch_size'}       # Dynamic batch dimension for output
    }

    torch.onnx.export(
        model,
        dummy_input,
        f=TMPL_ONNX_FILE_NAME.format(outlier_class.__name__, model_conf.image_size[0]),
        export_params=True,
        input_names=['img_grey'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes
    )


def print_details(outlier_class: type[ImgClassifier]):
    session = ort.InferenceSession(TMPL_ONNX_FILE_NAME.format(outlier_class.__name__, ModelConf.image_size[0]))
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
    export_to_onnx(ImgClassifier)
    print_details(ImgClassifier)
