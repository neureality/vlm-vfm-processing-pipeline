from vfm import VFM
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
dtype = torch.bfloat16
device = "cuda"
model = VFM(dtype=dtype)  # TODO: Fix the dtype issue when u need to specify it twice
model = model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

all_pixel_values = torch.load(
    "/home/odedh/nr_value_prop/submodules/vfm/test_data/all_pixel_values.pkl",
    weights_only=True,
    map_location="cuda",
).to(torch.float32)
patch_attn_mask = torch.load(
    "/home/odedh/nr_value_prop/submodules/vfm/test_data/patch_attn_mask.pkl",
    weights_only=True,
    map_location="cuda",
)

with torch.inference_mode():
    vision_embedding = model(
        all_pixel_values,
        patch_attn_mask,
        # tgt_sizes
    )
pass
