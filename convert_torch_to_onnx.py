import torch
from vfm import VFM
from scripts.torch_to_onnx import fix_onnx_fp16, convert_pytorch_to_onnx

OPSET_VERSION = 11
device = "cpu"
dtype = torch.float32
# Instantiate the model
model = VFM(device=device, dtype=dtype)
model.eval()  # Set model to evaluation mode

vision_embedding = torch.load(
    "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/vision_embedding_only_siglip.pt",
    weights_only=True,
    map_location=device,
).to(dtype)

# Set dynamic axes for inputs/outputs
dynamic_axes = {
    "vision_embedding": {0: "batch_size"},
    "vision_embedding_out": {0: "batch_size"},
}

convert_pytorch_to_onnx(
    model=model,
    input_sample=(vision_embedding,),
    onnx_path="models/vfm_only_resampler.onnx",
    input_names=[
        "vision_embedding",
    ],
    output_names=["vision_embedding_out"],
    dynamic_axes=dynamic_axes,
    opset_version=OPSET_VERSION,  # Higher opset for newer operators
    export_params=True,  # Include model weights
    do_constant_folding=True,  # Optimize constant operations
    verbose=True,  # Show detailed conversion info
    training=torch.onnx.TrainingMode.EVAL,  # Export in inference mode
    enable_onnx_checker=True,  # Verify model structure
    optimize_onnx=False,  # Apply optimizations
)


# Fix the ONNX model for mixed precision
fp16_model_name = fix_onnx_fp16(
    gen_models_path="models",
    model_base_name="vfm_only_resampler",
)
