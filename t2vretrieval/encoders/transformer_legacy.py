"""
Transformer implementation.

Similar inference speed to pytorch built-in transformers.
"""
import numpy as np
import torch as th
from torch import nn
from t2vretrieval.nntrainer.initialization import init_network
from t2vretrieval.nntrainer.encoder import PositionalEncodingSinCos
from t2vretrieval.nntrainer.mlp import MLP
from t2vretrieval.nntrainer.normalizations import LayerNormalization
from t2vretrieval.nntrainer.poolers import MultiGenPool

INF = 32752

# ---------- Module Implementations ----------

class TransformerLegacy(nn.Module):
    """
    The COOT transformer (In total there are 4 of these.)
    """

    def __init__(self, input_dim, output_dim, ispooling = True, inputreshape = True):
        super().__init__()
        self.input_dim = input_dim
        self.inputreshape = inputreshape
        # normalize input
        self.norm_input = LayerNormalization(self.input_dim)

        # convert input with FC
        self.output_dim = output_dim
        if inputreshape:
            self.input_fc = MLP(input_dim, output_dim)

        # embed time information
        self.embedding = PositionalEncodingSinCos(output_dim)

        # self-attention transformer
        self.tf = TransformerEncoder()

        # # subspace disabled for now
        # self.tf = SubspaceTransformerEncoder(
        #     cfg.num_layers, input_dim, cfg.num_heads,
        #     cfg.pointwise_ff_dim,
        #     cfg.dropout, cfg.activation, cfg.atn_ss_subspace,
        #     cfg.atn_ss_kernels, use_cuda=use_cuda)

        # build pooler
        self.ispooling = ispooling
        if ispooling:
            self.pooler = MultiGenPool(1, output_dim, 768, 2, 0.025)

        # correct current input dim, depending on pooler, pooler_heads = 2
        output_dim *= 2
        self.output_dim = output_dim

        # run the initializer
        init_network(self, "truncnorm", 0.01)

    def calculate_output_size(self) -> int:
        """
        Calculate output feature dim of this transformer model

        Returns:
            Output feature dim.
        """
        output_dim = self.output_dim
        return output_dim

    def forward(self, features, mask, lengths):
        """
        COOT forward pass. This is used in RetrievalModelManager to compute the embeddings.

        Args:
            features: Input features with shape (batch_size, max_seq_len, dim_features)
            mask: Mask with 0 for real data, 1 for padded elements to be ignored. Shape (batch_size, max_seq_len)
            lengths: Sequence length per datapoint (must correspond to the mask) shape (batch_size)
            hidden_state: Optional hidden state for cross-attention with shape (batch_size, dim_hidden)

        Returns:
            Tuple of:
                Features after pooling with shape (batch_size, dim_output)
                Features before pooling with shape (batch_size, max_seq_len, dim_hidden)
        """
        # print("ATN IN: feat",features.shape,"mask",mask.shape)
        # (batch, seq, input_dim)

        # normalize input
        features = self.norm_input(features)

        # convert input with FC
        if self.inputreshape:
            features = self.input_fc(features)
        # print("input fc",features.shape)
        # (batch, seq, new_dim)

        # add temporal encoding
        features = self.embedding(features)
        # print("embedding", features.shape)
        # (batch, seq, new_dim)

        # apply transformer
        features = self.tf(features, features, features, mask)
        # print("after transformer", features.shape, len(atns), atns[0].shape)

        # apply pooling
        if self.ispooling:
            pooled = self.pooler(features, mask, lengths)
            return pooled, features
        else:
            return features


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder.
    """

    def __init__(self):
        super().__init__()
        self.encoder_layers = TransformerEncoderLayer(384, 8, 384, 0.025)#drop_out_pro will change for different datasets

    def forward(self, query, key, value, mask):
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, key_len, d_model)
            mask: (batch_size, key_len)

        Returns:
            output (batch_size, query_len, d_model)
        """
        batch_size, query_len, _embed_dim = query.shape
        batch_size, key_len, _embed_dim = key.shape
        # for this transformer architecture, mask needs to be expanded

        # NOT squared instead of expanded:
        # having entire rows be zero means the softmax will be uniform.
        # the query will never attend to masked keys, and useless masked query
        # parts can be discarded on the output pool

        mask_expanded = mask.unsqueeze(1).expand(batch_size, query_len, key_len)
        # print(mask_expanded.shape)
        # (batch_size, query_len, key_len) dtype bool

        sources = self.encoder_layers(query, key, value, mask_expanded)
        return sources


class TransformerEncoderLayer(nn.Module):
    """
    Self Attention Layer as in BERT.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_prob = 0.):
        super().__init__()
        self.self_attention_layer = Sublayer(MultiHeadAttention(num_heads, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, key_len, d_model)
            sources_mask: (batch_size, query_len, key_len)

        Returns:
            output: (batch_size, query_len, d_model)
        """
        # sources: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)

        return sources


class Sublayer(nn.Module):
    """
    Add Residual and Layernorm to the given layer.
    """

    def __init__(self, sublayer, d_model):
        super().__init__()

        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        """
        Sublayer forward.
        """
        # save input for residual
        x = args[0]
        # run sublayer
        sublayer_return = self.sublayer(*args)
        x = sublayer_return + x
        return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention.
    """

    def __init__(self, num_heads, d_model, dropout_prob):
        super().__init__()
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.query_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.key_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.value_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.final_projection = nn.Linear(d_model, num_heads * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask_expanded = None, _layer_cache=None):
        """
        value_len must be equal to key_len
        query_len is the output length

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, key_len_len, model_dim)
            mask_expanded: (batch_size, query_len, key_len)
            _layer_cache: DecoderState (unused)

        Returns:
            output: (batch_size, query_len, model_dim)
        """
        # print("attention mask", mask)
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.num_heads

        query_projected = self.query_projection(query)  # shape (batch_size, query_len, num_heads, d_head)
        key_projected = self.key_projection(key)  # shape (batch_size, key_len, num_heads, d_head)
        value_projected = self.value_projection(value)  # shape (batch_size, key_len, num_heads, d_head)

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.num_heads, d_head).transpose(1, 2)
        # print("query_heads", query_heads.shape)
        # (batch_size, num_heads, query_len, d_head)

        key_heads = key_projected.view(batch_size, key_len, self.num_heads, d_head).transpose(1, 2)
        # print("key_heads", key_heads.shape)
        # (batch_size, num_heads, key_len, d_head)

        value_heads = value_projected.view(batch_size, value_len, self.num_heads, d_head).transpose(1, 2)
        # print("value_heads", value_heads.shape)
        # (batch_size, num_heads, key_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads, key_heads)
        # print("attention_weights", attention_weights.shape)
        # (batch_size, num_heads, query_len, key_len)

        if mask_expanded is not None:
            mask_expanded_per_head = mask_expanded.unsqueeze(1).expand_as(attention_weights)
            # print("mask_expanded_per_head", mask_expanded_per_head.shape)
            # shape (batch_size, num_heads, query_len, key_len)
            attention_weights = attention_weights.masked_fill(mask_expanded_per_head, INF)
            # print("attention_weights", attention_weights.shape)
            # shape (batch_size, num_heads, query_len, query_len)

        # DONT Save attention to the object
        attention = self.softmax(attention_weights)
        # print("attention_weights", attention_weights.shape)

        attention_dropped = self.dropout(attention)
        context_heads = th.matmul(attention_dropped, value_heads)
        # shape (batch_size, num_heads, query_len, d_head)
        # print("context_heads", context_heads.shape)

        context_sequence = context_heads.transpose(1, 2)
        # (batch_size, query_len, num_heads, d_head)

        context = context_sequence.reshape(batch_size, query_len, d_model)
        # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        # print("final_output", final_output.shape)

        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """
        Args:
             query_heads: (batch_size, num_heads, query_len, d_head)
             key_heads: (batch_size, num_heads, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = th.matmul(query_heads, key_heads_transposed)
        # (batch_size, num_heads, query_len, key_len)

        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    """
    Feedforward on last dimension (pointwise) with default activation Relu
    and DropOut.
    """

    def __init__(self, d_ff, d_model, dropout_prob):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)
