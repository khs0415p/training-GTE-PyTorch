""" GTE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GTEConfig(PretrainedConfig):
    model_type = "gte"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.,
        max_position_embeddings=4096,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        layer_norm_type="layer_norm",
        pack_qkv=True,
        pad_token_id=0,
        position_embedding_type="rope",
        rope_scaling={"factor":2.0, "type": "ntk"},
        rope_theta=500000,
        torch_dtype="float32",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.pack_qkv = pack_qkv
        self.torch_dtype = torch_dtype
        self.position_embedding_type = position_embedding_type
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

if __name__ == "__main__":
    config = GTEConfig()
    print(config.use_return_dict)
    print(config.output_attentions)
    print(config.output_hidden_states)
    print(config.is_decoder)