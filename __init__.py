"""SigLIP2 image/text embeddings exported as JSON for ComfyUI API consumers."""

import json
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()
_MODELS = [
    "google/siglip2-base-patch16-256",
    "google/siglip2-so400m-patch14-384",
]


def _load(model_id):
    with _MODEL_LOCK:
        if model_id not in _MODEL_CACHE:
            from transformers import AutoModel, AutoProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
            model = model.to(device).eval()
            _MODEL_CACHE[model_id] = (model, processor, device)
        return _MODEL_CACHE[model_id]


def _extract_features(output):
    features = getattr(output, "pooler_output", None)
    if features is None:
        if isinstance(output, torch.Tensor):
            features = output
        elif isinstance(output, (tuple, list)):
            features = output[0]
    if features is None or features.ndim != 2:
        raise RuntimeError("SigLIP2 returned unexpected features shape")
    return F.normalize(features.float(), p=2, dim=-1).cpu().tolist()


def _json_response(vectors, model_id, kind):
    result = json.dumps(
        {"model": model_id, "type": kind, "count": len(vectors), "vectors": vectors},
        ensure_ascii=False, separators=(",", ":"),
    )
    return {"ui": {"text": [result]}, "result": (result,)}


class SigLIP2ImageJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "model_id": (_MODELS, {"default": _MODELS[0]}),
            "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "encode"
    CATEGORY = "Art Explorer/SigLIP2"
    OUTPUT_NODE = True

    def encode(self, images, model_id, batch_size):
        model, processor, device = _load(model_id)
        vectors = []
        for start in range(0, images.shape[0], batch_size):
            batch = images[start:start + batch_size].detach().cpu().numpy()
            pil_images = [Image.fromarray(np.uint8(np.clip(img * 255, 0, 255))).convert("RGB") for img in batch]
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
            vectors.extend(_extract_features(features))
        return _json_response(vectors, model_id, "image")


class SigLIP2TextJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "queries": ("STRING", {"multiline": True, "default": '["рыжий кот", "олень"]'}),
            "model_id": (_MODELS, {"default": _MODELS[0]}),
            "batch_size": ("INT", {"default": 16, "min": 1, "max": 128}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "encode"
    CATEGORY = "Art Explorer/SigLIP2"
    OUTPUT_NODE = True

    def encode(self, queries, model_id, batch_size):
        try:
            texts = json.loads(queries)
        except json.JSONDecodeError:
            texts = [line.strip() for line in queries.splitlines() if line.strip()]
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("queries must be a non-empty JSON list of strings (or newline-separated strings)")
        model, processor, device = _load(model_id)
        vectors = []
        for start in range(0, len(texts), batch_size):
            inputs = processor(text=texts[start:start + batch_size], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                features = model.get_text_features(**inputs)
            vectors.extend(_extract_features(features))
        return _json_response(vectors, model_id, "text")


NODE_CLASS_MAPPINGS = {
    "ArtExplorerSigLIP2ImageJSON": SigLIP2ImageJSON,
    "ArtExplorerSigLIP2TextJSON": SigLIP2TextJSON,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtExplorerSigLIP2ImageJSON": "SigLIP2 Image → JSON",
    "ArtExplorerSigLIP2TextJSON": "SigLIP2 Text → JSON",
}
