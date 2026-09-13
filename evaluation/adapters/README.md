# evaluation/adapters

Each adapter wraps one model family behind the common `ModelAdapter`
interface (`load`, `predict_dataset`, `count_params`, `estimate_gflops`,
`measure_speed`), so `evaluate.py` can score any of them the same way.

| File | Model |
|------|-------|
| `base.py` | The `ModelAdapter` abstract interface every adapter implements. |
| `data.py` | Dataset pairing/splitting helpers shared by the adapters below. |
| `segformer.py` | Vanilla SegFormer-B0 and the Attention Consistency variant, delegating to `attention_consistency/` for the model and Grad-Rollout attention map. |
| `unet_torch.py` | The PyTorch U-Net baseline (`unet_model.py` at the repo root). |
| `unet_keras.py` | A legacy Keras U-Net path, superseded by `unet_torch.py`; needs `tensorflow` (optional, see `requirements.txt`). |
| `deeplab_model.py` | The DeepLabV3+ (MobileNetV3-Large) model builder. |
| `deeplabv3.py` | The DeepLabV3+ adapter, built on `deeplab_model.py`. |
| `segformer_stub.py` | A stub adapter that raises `NotImplementedError` on attention-related calls; a placeholder for a code path not wired up in this harness. |
