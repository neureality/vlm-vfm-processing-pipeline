from vfm import VFM
import torch
import os
from constants import DEVICE

if DEVICE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model = VFM()
model = model.to(device=DEVICE, dtype=torch.bfloat16)
model.eval()

all_pixel_values = torch.load(
    # "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/all_pixel_values.pkl",
    "/work/ronliv/ws/vlm_video_processing_pipeline/outputs/all_pixel_values.pkl",
    weights_only=True,
    map_location=DEVICE,
)
patch_attn_mask = torch.load(
    # "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/patch_attn_mask.pkl",
    "/work/ronliv/ws/vlm_video_processing_pipeline/outputs/patch_attn_mask.pkl",
    weights_only=True,
    map_location=DEVICE,
)
tgt_sizes = torch.load(
    # "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/tgt_sizes.pkl",
    "/work/ronliv/ws/vlm_video_processing_pipeline/outputs/tgt_sizes.pkl",
    weights_only=True,
    map_location=DEVICE,
)

with torch.inference_mode():
    vision_embedding = model(
        all_pixel_values,
        patch_attn_mask,
        # tgt_sizes
        )
pass
