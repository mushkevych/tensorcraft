from os import path

import torch
import torch_neuronx
from torch import nn

from mlpbertproj.classifier.mlpbert_classifier import MlpBertModel
from mlpbertproj.classifier.mlpbert_configuration import ModelConf
from mlpbertproj.classifier.onnx_exporter import WEIGHTS_FILE_NAME
from utils.lm_components import LmComponents, DSMODELS_PREFIX, FQFP_MODEL_BERT_BASE, FQFP_MODEL_BERT_BASE_XLA, \
    GraphCodeBertConf

# XLA stands for Accelerated Linear Algebra
FQFP_WEIGHTS_SEC_CPU = path.join(DSMODELS_PREFIX, 'tensorcraft', 'mlpbert_classifier', WEIGHTS_FILE_NAME)
FQFP_MODEL_SEC_XLA = path.join(DSMODELS_PREFIX, 'tensorcraft', 'mlpbert_classifier', f'{MlpBertModel.__name__}.xlapt')

model_conf = ModelConf()


class BertModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract only `last_hidden_state` from the model's output
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        return last_hidden_state


def _xla_model_compilation(model: nn.Module, example_inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
    traced_module = torch_neuronx.trace(
        model,
        example_inputs=example_inputs,
        compiler_args=[
            '--target=inf2',
            '--auto-cast=matmult',
            '--auto-cast-type=fp16',
            '--enable-mixed-precision-accumulation',
            '--enable-fast-loading-neuron-binaries',
        ]
    )
    return traced_module


def export_bert_to_xla():
    # Load the model and tokenizer
    ml_components = LmComponents(model_name=FQFP_MODEL_BERT_BASE, device_name='cpu')
    ml_components.load(output_hidden_states=True)

    # Example input for model tracing
    max_length = GraphCodeBertConf.max_position_embeddings
    example_inputs = ml_components.tokenizer(
        'Sample input for embedding', return_tensors='pt', max_length=max_length, truncation=True, padding='max_length'
    )

    model_traced = _xla_model_compilation(
        BertModelWrapper(ml_components.llm),
        example_inputs=(example_inputs['input_ids'], example_inputs['attention_mask'])
    )

    # Save the compiled NeuronX model
    model_traced.save(FQFP_MODEL_BERT_BASE_XLA)


def export_sec_to_xla():
    model = MlpBertModel(model_conf)
    model.load_model_weights(FQFP_WEIGHTS_SEC_CPU)

    # Example input for model tracing
    example_inputs = torch.randn(size=(1, model_conf.input_size))  # 1 stands for batch dimension
    traced_script_module = _xla_model_compilation(model, example_inputs=example_inputs)

    # Save the compiled NeuronX model
    traced_script_module.save(FQFP_MODEL_SEC_XLA)


def main():
    export_bert_to_xla()
    export_sec_to_xla()


if __name__ == '__main__':
    main()
