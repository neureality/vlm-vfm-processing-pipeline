import torch
from vfm import VFM
from scripts.torch_to_onnx import fix_onnx_fp16, convert_pytorch_to_onnx

IS_FOR_QPC_BUILD = True

BS = 30  # FIXME: also change BS in the siglip config file
model_base_name = f"vfm_{'f32' if IS_FOR_QPC_BUILD else 'bf16'}_BS{BS}_tomasz"

# ONNX opset version 15 or higher to properly support bfloat16 (move back to 13 when QPC)
OPSET_VERSION = 13 if IS_FOR_QPC_BUILD else 15
device = "cpu" if IS_FOR_QPC_BUILD else "cuda"  # Use CPU for QPC build
# Use float32 for QPC build
dtype = torch.float32 if IS_FOR_QPC_BUILD else torch.bfloat16
# Instantiate the model
# FIXME: Fix the dtype issue when u need to specify it twice
model = VFM(dtype=dtype, device=device, batch_size=BS)
model = model.to(device=device, dtype=dtype)
model.eval()

all_pixel_values = torch.load(
    "/home/odedh/nr_value_prop/submodules/vfm/test_data/all_pixel_values.pkl",
    # "/home/odedh/nr_value_prop/all_pixel_values_ski.pth",
    weights_only=True,
    map_location=torch.device(device),
).to(torch.float32)
all_pixel_values = all_pixel_values.to(dtype=torch.float32) if IS_FOR_QPC_BUILD else all_pixel_values

patch_attn_mask = torch.load(
    "/home/odedh/nr_value_prop/submodules/vfm/test_data/patch_attn_mask.pkl",
    # "/home/odedh/nr_value_prop/ski_patch_attn_mask.pth",
    weights_only=True,
    map_location=torch.device(device),
)

# FIXME: Its a tmp
all_pixel_values = torch.concat(
    [all_pixel_values, all_pixel_values, all_pixel_values])
patch_attn_mask = torch.concat(
    [patch_attn_mask, patch_attn_mask, patch_attn_mask])

# Setting the BS
all_pixel_values = all_pixel_values[:BS, ...]  # Select a batch size
patch_attn_mask = patch_attn_mask[:BS, ...]  # Select a batch size
# FIXME: Duplicate over first domention to create BS > Original BS

# Set dynamic axes for inputs/outputs
dynamic_axes = {
    "all_pixel_values": {0: "batch_size"},
    "patch_attn_mask": {0: "batch_size"},
    "vision_embedding": {0: "batch_size"},
}
try:
    convert_pytorch_to_onnx(
        model=model,
        input_sample=(
            all_pixel_values,
            patch_attn_mask,
        ),
        onnx_path=f"models/{model_base_name}.onnx",
        input_names=[
            "all_pixel_values",
            "patch_attn_mask",
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
except Exception as e:
    print(f"Error during conversion: {e}")
    raise

finally:
    # Fix the ONNX model for mixed precision
    print("Fixing ONNX model for mixed precision...")
    fp16_model_name = fix_onnx_fp16(
        gen_models_path="models",
        model_base_name=model_base_name,
    )
