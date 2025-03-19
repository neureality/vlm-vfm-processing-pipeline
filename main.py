from vfm import VFM
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model = VFM()
model = model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

vision_embedding = torch.load(
    "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/vision_embedding_only_siglip.pt",
    weights_only=True,
    map_location="cuda",
).to(torch.float32)

with torch.inference_mode():
    vision_embedding = model(
        vision_embedding,
    )
pass
