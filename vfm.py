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
        # dtype: torch.dtype = torch.bfloat16,
        # dtype: torch.dtype = torch.float32,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.siglip_config = AutoConfig.from_pretrained(siglip_config)
        self.resampler_config = AutoConfig.from_pretrained(resampler_config)
        # Pre Compute tgt_sizes 🌵
        pre_computed_tgt_sizes = self._pre_compute_tgt_sizes(self.siglip_config)
        self.register_buffer("pre_computed_tgt_sizes", pre_computed_tgt_sizes)
        
        self.vpm = self.init_vision_module()
        self.resampler = self.init_resampler()
        self.load_model_weights()


    def init_vision_module(self) -> nn.Module:
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.siglip_config._attn_implementation == "flash_attention_2":
            self.siglip_config._attn_implementation = "flash_attention_2"
        else:
            # not suport sdpa
            self.siglip_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(
            self.siglip_config,
            self.pre_computed_tgt_sizes
            )
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
            pre_computed_tgt_sizes=self.pre_computed_tgt_sizes,
        )
        return resampler


    def _pre_compute_tgt_sizes(self, config):
        # 1. Parse relevant config entries
        batch_size = config.batch_size                # e.g. 30
        patch_size = config.patch_size                # e.g. 14
        resize_global = config.resize_global          # e.g. [336, 602]
        resize_refine = config.resize_refine          # e.g. [476, 840]
        grid = config.crop_best_grid                  # e.g. [1, 2]

        # 2. Compute the global slice’s patch dimensions
        global_h, global_w = resize_global
        global_tgt = (global_h // patch_size, global_w // patch_size)
        # e.g. (336 // 14, 602 // 14) -> (24, 43)

        # 3. Compute how many refine patches we create per frame
        #    For grid=[1, 2], we produce 1 row * 2 columns = 2 refine patches per frame
        refine_h, refine_w = resize_refine
        patch_h = refine_h // grid[0]   # e.g. 476 // 1 = 476
        patch_w = refine_w // grid[1]   # e.g. 840 // 2 = 420
        refine_tgt = (patch_h // patch_size, patch_w // patch_size)
        # e.g. (476 // 14, 420 // 14) -> (34, 30)

        # 4. Each frame => 1 global slice + grid[0]*grid[1] refine slices
        #    For each slice, we store the patch dimension as a (H//patch_size, W//patch_size)
        num_refine_slices = grid[0] * grid[1]  # e.g. 2
        slices_per_frame = 1 + num_refine_slices  # e.g. 3 total slices per frame

        # 5. Build the per-frame pattern, then repeat for batch_size frames
        frame_pattern = [global_tgt] + [refine_tgt] * num_refine_slices
        # e.g. [ (24,43), (34,30), (34,30) ]
        repeated_list = frame_pattern * (batch_size // slices_per_frame)
        # e.g. repeated_list has length 3 * (30/3) = 30 if batch_size=30 and slices_per_frame=3

        # 6. Convert to a PyTorch tensor on the desired device
        #    dtype can be int32 or int64, depending on your usage
        tgt_sizes = torch.tensor(repeated_list, dtype=torch.int64)
        
        return tgt_sizes

    def load_model_weights(
        self,
        resampler_state_dict_path: str = RESAMPLER_STATE_DICT_PATH,
        vpm_state_dict_path: str = SIGLIP_STATE_DICT_PATH,
    ) -> None:
        self.resampler.load_state_dict(
            torch.load(resampler_state_dict_path, weights_only=True),
            strict=False # ignore missing keys 🌵
        )
        self.vpm.load_state_dict(
            torch.load(vpm_state_dict_path, weights_only=True),
            strict=False # ignore missing keys 🌵
            ) 

    def forward(
        self,
        all_pixel_values: torch.Tensor,
        patch_attn_mask: torch.Tensor,
        # tgt_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if all_pixel_values.dtype != self.dtype:
            # print(
            #     f"Current dtype of all_pixel_values is {all_pixel_values.dtype}, Converting all_pixel_values to {self.dtype}"
            # )
            all_pixel_values = all_pixel_values.to(self.dtype)

        vision_embedding = self.vpm(
            all_pixel_values, patch_attention_mask=patch_attn_mask
        ).last_hidden_state
        print("vision_embedding.dtype", vision_embedding.dtype)
        vision_embedding = self.resampler(vision_embedding)

        return vision_embedding
