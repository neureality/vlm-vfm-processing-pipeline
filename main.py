from vfm import VFM
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model = VFM()
model = model.to(device="cuda", dtype=torch.bfloat16)

all_pixel_values = torch.load("/home/ubuntu/vlm-vfm-processing-pipeline/test_data/all_pixel_values.pkl", weights_only=True, map_location="cuda")
patch_attn_mask = torch.load("/home/ubuntu/vlm-vfm-processing-pipeline/test_data/patch_attn_mask.pkl", weights_only=True, map_location="cuda")
tgt_sizes = torch.load("/home/ubuntu/vlm-vfm-processing-pipeline/test_data/tgt_sizes.pkl", weights_only=True, map_location="cuda")

with torch.inference_mode():
    vision_embedding = model(all_pixel_values, patch_attn_mask, tgt_sizes)
pass