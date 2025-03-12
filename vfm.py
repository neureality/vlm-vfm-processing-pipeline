from torch import nn
import torch
from resampler.resampler import Resampler2_5
from siglip.idefics2_vision_model import Idefics2VisionTransformer
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


class VFM(nn.Module):
    def __init__(self, siglip_config, resampler_config):
        super().__init__()
        self.siglip_config = siglip_config
        self.resampler_config = resampler_config
        self.resampler = self.init_resampler()
        self.vpm = self.init_vision_module()

    def init_vision_module(self) -> nn.Module:
        model = Idefics2VisionTransformer(self.siglip_config)
        if self.siglip_config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self, embed_dim: int, vision_dim: int) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(
                num_queries=self.resampler_config.query_num,
                embed_dim=self.resampler_config.embed_dim,
                num_heads=self.resampler_config.embed_dim // 128,
                kv_dim=self.resampler_config.vision_dim,
            )
        return resampler
