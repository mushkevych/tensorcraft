from os import path
from typing import Literal

from utils.lm_components import MODEL_BERT_BASE, LmComponents, FQFP_MODEL_FASTTEXT_D32, load_ft_model, fasttext


def load_hf_token() -> str | None:
    fqfp_token = path.abspath(path.join(path.dirname(__file__), '..', '.hugging_face.token'))
    if not path.exists(fqfp_token):
        return None

    with open(fqfp_token, 'r') as f:
        token = f.read().strip()
        # print(token)
        return token



def instantiate_ml_components(
    model_name: str = MODEL_BERT_BASE,
    device_name: str | None = None,
    compile_model: bool = False,
    model_mode: Literal['eval', 'train'] = 'eval'
) -> LmComponents:
    """
    Lazily create and cache LmComponents by model_name.
    A dict[str, LmComponents] is stored on this function as `.cache`.
    NOTE: to clear the cache, call: `getattr(instantiate_ml_components, "cache", {}).clear()`
    """
    # Initialize the per-function cache dict if missing
    cache: dict[str, LmComponents] = getattr(instantiate_ml_components, 'cache', None)
    if cache is None:
        cache = {}
        setattr(instantiate_ml_components, 'cache', cache)

    # Instantiate on first request for this model_name, else return cached
    if model_name not in cache:
        lm = LmComponents(model_name=model_name, device_name=device_name)
        lm.load(
            hf_token=load_hf_token(),
            output_hidden_states=True,
            compile_model=compile_model,
            model_mode=model_mode,
        )
        cache[model_name] = lm

    return cache[model_name]


def instantiate_ft(model_name: str = FQFP_MODEL_FASTTEXT_D32) -> fasttext.FastText:
    """
    Lazily create and cache FastText models by model_name.
    A dict[str, fasttext.FastText] is stored on this function as `.cache`.
    NOTE: to clear the cache, call: `getattr(instantiate_ft, "cache", {}).clear()`
    """
    cache: dict[str, fasttext.FastText] = getattr(instantiate_ft, 'cache', None)
    if cache is None:
        cache = {}
        setattr(instantiate_ft, 'cache', cache)

    if model_name not in cache:
        cache[model_name] = load_ft_model(model_name)

    return cache[model_name]
