import torch
from torch import nn
from transformers import AutoConfig
from constants import *

from siglip.modeling_navit_siglip import SiglipVisionTransformer
from resampler.resampler import Resampler


class VFM(nn.Module):
    def __init__(
        self,
        siglip_config: str = SIGLIP_CONFIG_FILE_PATH,
        resampler_config: str = RESAMPLER_CONFIG_FILE_PATH,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.siglip_config = AutoConfig.from_pretrained(siglip_config)
        self.resampler_config = AutoConfig.from_pretrained(resampler_config)
        self.resampler = self.init_resampler()
        self.vpm = self.init_vision_module()

        self._load_model_weights()

    def init_vision_module(self) -> nn.Module:
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.siglip_config._attn_implementation == "flash_attention_2":
            self.siglip_config._attn_implementation = "flash_attention_2"
        else:
            # not suport sdpa
            self.siglip_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.siglip_config)
        if self.siglip_config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self) -> nn.Module:
        # The resampler in 2.6 remains consistent with the one in 2.5.
        resampler = Resampler(
            num_queries=self.resampler_config.query_num,
            embed_dim=self.resampler_config.embed_dim,
            num_heads=self.resampler_config.embed_dim // 128,
            kv_dim=self.resampler_config.vision_dim,
            adaptive=True,
        )
        return resampler

    def _load_model_weights(
        self,
        resampler_state_dict_path: str = RESAMPLER_STATE_DICT_PATH,
        vpm_state_dict_path: str = SIGLIP_STATE_DICT_PATH,
    ) -> None:
        self.resampler.load_state_dict(
            torch.load(resampler_state_dict_path, weights_only=True)
        )
        self.vpm.load_state_dict(
            torch.load(vpm_state_dict_path, weights_only=True),
            strict=False # ignore missing keys -> tgt_sizes registered buffer is missing 🌵
            ) 

    def forward(
        self,
        all_pixel_values: torch.Tensor,
        patch_attn_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if all_pixel_values.dtype != self.dtype:
            print(
                f"Current dtype of all_pixel_values is {all_pixel_values.dtype}, Converting all_pixel_values to {self.dtype}"
            )
            all_pixel_values = all_pixel_values.to(self.dtype)

        vision_embedding = self.vpm(
            all_pixel_values, patch_attention_mask=patch_attn_mask
        ).last_hidden_state
        vision_embedding = self.resampler(vision_embedding, tgt_sizes)

        return vision_embedding
