from popgym.baselines.ray_models.base_model import BaseModel
from torch import nn
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import torch
from typing import List, Tuple
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, Optional, Union

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules import (
    GRUGate,
    RelativeMultiHeadAttention,
    SkipConnection,
)
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
torch, nn = try_import_torch()


class GTrXLModel(BaseModel):

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,

    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.num_transformer_units = 1
        self.attention_dim = 128
        self.num_heads = 2
        self.head_dim = 64
        self.max_seq_len = model_config["max_seq_len"]
        self.obs_dim = obs_space.shape[0]
        self.position_wise_mlp_dim = 64
        self.init_gru_gate_bias = 2.0
        linear_layer = nn.Identity()

        layers = [linear_layer]

        attention_layers = []
        for i in range(self.num_transformer_units):

            MHA_layer = SkipConnection(
                RelativeMultiHeadAttention(
                    in_dim=self.attention_dim,
                    out_dim=self.attention_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    input_layernorm=True,
                    output_activation=nn.LeakyReLU,
                ),
                fan_in_layer=GRUGate(self.attention_dim,
                                     self.init_gru_gate_bias),
            )

            E_layer = SkipConnection(
                nn.Sequential(
                    torch.nn.LayerNorm(self.attention_dim),
                    SlimFC(
                        in_size=self.attention_dim,
                        out_size=self.position_wise_mlp_dim,
                        use_bias=False,
                        activation_fn=nn.LeakyReLU,
                    ),
                    SlimFC(
                        in_size=self.position_wise_mlp_dim,
                        out_size=self.attention_dim,
                        use_bias=False,
                        activation_fn=nn.LeakyReLU,
                    ),
                ),
                fan_in_layer=GRUGate(self.attention_dim,
                                     self.init_gru_gate_bias),
            )

            attention_layers.extend([MHA_layer, E_layer])

        final_attention_layers = nn.Sequential(*attention_layers)
        layers.extend(attention_layers)

        self.core = nn.ModuleList(layers)

    def initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(
                1, self.attention_dim
            )
            for i in range(self.num_transformer_units)
        ]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        all_out = z
        memory_outs = []
        for i in range(len(self.core)):
            if i % 2 == 1:
                all_out = self.core[i](all_out, memory=state[i // 2])
            else:
                all_out = self.core[i](all_out)
                memory_outs.append(all_out)

        memory_outs = memory_outs[:-1]
        z = all_out
        state = memory_outs

        return z, state
