import torch
from vfm import VFM
from scripts.torch_to_onnx import fix_onnx_fp16, convert_pytorch_to_onnx

OPSET_VERSION = 11
device = "cpu"
dtype = torch.float32
# Instantiate the model
model = VFM(device=device, dtype=dtype)
model.eval()  # Set model to evaluation mode

all_pixel_values = torch.load(
    "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/all_pixel_values.pkl",
    weights_only=True,
    map_location=device,
)
patch_attn_mask = torch.load(
    "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/patch_attn_mask.pkl",
    weights_only=True,
    map_location=device,
)
tgt_sizes = torch.load(
    "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/tgt_sizes.pkl",
    weights_only=True,
    map_location=device,
)

# Set dynamic axes for inputs/outputs
dynamic_axes = {
    "all_pixel_values": {0: "batch_size"},
    "patch_attn_mask": {0: "batch_size"},
    # "tgt_sizes": {0: "batch_size"},
    "vision_embedding": {0: "batch_size"},
}

convert_pytorch_to_onnx(
    model=model,
    input_sample=(
        all_pixel_values,
        patch_attn_mask,
        # tgt_sizes
        ),
    onnx_path="models/vfm.onnx",
    input_names=[
        "all_pixel_values",
        "patch_attn_mask",
        # "tgt_sizes"
        ],
    output_names=["vision_embedding"],
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
    model_base_name="vfm",
)