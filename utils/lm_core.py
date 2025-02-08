from os import path

from utils.lm_components import MODEL_BERT_BASE, LmComponents, FQFP_MODEL_FASTTEXT_D32, load_ft_model, fasttext


def load_hf_token() -> str:
    fqfp_token = path.abspath(path.join(path.dirname(__file__), '..', '.hugging_face.token'))
    with open(fqfp_token, 'r') as f:
        token = f.read().strip()
        # print(token)
        return token


# Global variables for multi-processing
_ml_components = None
_ft_model = None


def instantiate_ml_components(model_name: str = MODEL_BERT_BASE, device_name: str = None) -> LmComponents:
    global _ml_components
    if _ml_components is None:
        # Only instantiate if it hasn't been done in this process
        _ml_components = LmComponents(model_name=model_name, device_name=device_name)
        _ml_components.load(hf_token=load_hf_token(), output_hidden_states=True)
    return _ml_components


def instantiate_ft(model_name: str = FQFP_MODEL_FASTTEXT_D32) -> fasttext.FastText:
    global _ft_model
    if _ft_model is None:
        # Only instantiate if it hasn't been done in this process
        _ft_model = load_ft_model(model_name)
    return _ft_model
