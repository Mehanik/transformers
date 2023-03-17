# coding=utf-8
# Copyright 2023 Evgeny Mikhantyev
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERGMAN configuration"""
from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BERGMAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class BergmanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BergmanModel`] or a [`TFBergmanModel`]. It is
    used to instantiate a BERGMAN model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Bergman model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BergmanModel`] or [`TFBergmanModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_matrix_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BergmanModel`] or [`TFBergmanModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import BergmanConfig, BergmanModel

    >>> # Initializing a Bergman configuration
    >>> configuration = BergmanConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = BergmanModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bergman"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_matrix_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        output_matrices=False,
        matrix_norm_alg=None,
        matrix_dim=16,
        vector_init_direction="one",
        use_for_context=["lr"],
        networks_for_heads=None,
        matrix_norm_loss_type=None,
        matrix_norm_loss_axis=(-1,),
        matrix_norm_loss_k=1.0,
        matrix_unitary_loss=None,
        matrix_unitary_loss_k=1.0,
        matrix_encoder_two_layers=False,
        matrix_norm_preheat_steps=0,
        norm_vectors=False,
        detach_norm_vectors=False,
        vector_norm_eps=1e-6,
        matrix_norm_eps=1e-6,
        complex_matrix=False,
        complex_matrix_abs=False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_matrix_heads = num_matrix_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.output_matrices = output_matrices
        self.matrix_norm_alg = matrix_norm_alg
        self.matrix_dim = matrix_dim
        self.vector_init_direction = vector_init_direction
        self.use_for_context = use_for_context
        self.networks_for_heads = networks_for_heads
        self.matrix_norm_loss_type = matrix_norm_loss_type
        self.matrix_norm_loss_k = matrix_norm_loss_k
        self.matrix_norm_loss_axis = matrix_norm_loss_axis
        self.matrix_encoder_two_layers = matrix_encoder_two_layers
        self.matrix_unitary_loss = matrix_unitary_loss
        self.matrix_unitary_loss_k = matrix_unitary_loss_k
        self.matrix_norm_preheat_steps = matrix_norm_preheat_steps
        self.norm_vectors = norm_vectors
        self.detach_norm_vectors = detach_norm_vectors
        self.vector_norm_eps = vector_norm_eps
        self.matrix_norm_eps = matrix_norm_eps
        self.complex_matrix = complex_matrix
        self.complex_matrix_abs = complex_matrix_abs
