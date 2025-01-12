## Tips for expanding OpenCLIP

1. Configuration files in `open_clip/src/open_clip/model_configs/xxx.json`
- e.g., `convnext_large_d_320-adapter.json`

2. Implement new module in `open_clip/src/open_clip/your-module.py`
- e.g., `clip_model_adapter.py`

3. Import and modify corresponding section in `open_clip/src/open_clip/factory.py`
4. Modify the `open_clip/src/open_clip/pretrained.py` to load pre-trained checkpoints.

