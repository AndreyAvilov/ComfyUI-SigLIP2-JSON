# ComfyUI SigLIP2 JSON embeddings

Two ComfyUI custom nodes for normalized image/text vectors in the same embedding space.

## Install

Extract the folder to `ComfyUI/custom_nodes/ComfyUI-SigLIP2-JSON`.
Install `requirements.txt` using the Python environment that runs ComfyUI, then restart ComfyUI.
The model weights download from Hugging Face on first execution (internet connection needed); alternatively pre-download them to the HF cache.

## Nodes

- **SigLIP2 Image → JSON**: standard ComfyUI `IMAGE` tensor `[B,H,W,C]` and a batch size.
- **SigLIP2 Text → JSON**: `STRING` containing a JSON list of text queries, e.g. `["кот", "олень"]`, or one query per line.

Both expose a normal `STRING` output and also return JSON through the node `ui.text` field in the ComfyUI API `/history/{prompt_id}` result. No `ShowText` node is required (but one can be connected).

JSON shape:

```json
{"model":"google/siglip2-base-patch16-256","type":"image","count":1,"vectors":[[0.01,0.02]]}
```

Embeddings are L2 normalized. Compare vectors with their dot product (cosine similarity). **Use the same model ID on both nodes.**

**Note:** The `Anima SigLIP2 Encode Image` feature output for IP-Adapter is not the same API interface and should not be used as input here.

**Performance:** JSON with 100,000 vectors is too large to create in one prompt. Process small batches and immediately persist vectors in your own index/database.
