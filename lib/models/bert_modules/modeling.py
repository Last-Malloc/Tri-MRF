from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import math
import sys
import torch
from numpy import unicode
from torch import nn


def gelu(x):
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    """
    Bert  Multi-Head Attention
    """
    def __init__(self, seq_type, config):
        """

        :param seq_type: TXT/VIS
        :param config:
        """
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.query_other = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_other = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_other = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.seq_type = seq_type

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, hidden_states_other, attention_mask):
        """
        text查询text||visual 或 visual查询visual||text
        :param hidden_states: [batch_size, seq_length, hidden_size]
        :param hidden_states_other: [batch_size, seq_length_other, hidden_size]
        :param attention_mask: [batch_size, 1, 1, batch_max_len_sentence] can broadcast to[batch_size, num_heads, from_seq_length, batch_max_len_sentence]
        :return:
            context_layer [batch_size, seq_length, hidden_size]
        """
        # Q K V映射
        # query/key/value_layer other_query_layer [batch_size, num_attention_heads, seq_length, attention_head_size]
        # other_key/value_layer [batch_size, num_attention_heads, seq_length_other, attention_head_size]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        other_query_layer = self.query_other(hidden_states)
        other_key_layer = self.key_other(hidden_states_other)
        other_value_layer = self.value_other(hidden_states_other)

        other_query_layer = self.transpose_for_scores(other_query_layer)
        other_key_layer = self.transpose_for_scores(other_key_layer)
        other_value_layer = self.transpose_for_scores(other_value_layer)

        # 计算注意力分数 [batch_size, num_attention_heads, seq_length, seq_length] [batch_size, num_attention_heads, seq_length, seq_length_other]
        # 若当前seq_type为TXT(hidden_states为文本信息)，将遮盖加到attention_scores（对应text对text注意力分数）上
        # 若当前seq_type为VIS(hidden_states为文本信息)，将遮盖加到other_attention_scores（对应visual对text注意力分数）上
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        other_attention_scores = torch.matmul(other_query_layer, other_key_layer.transpose(-1, -2))
        other_attention_scores = other_attention_scores / math.sqrt(self.attention_head_size)

        if self.seq_type == 'TXT':
            attention_scores = attention_scores + attention_mask
        elif self.seq_type == 'VIS':
            other_attention_scores = other_attention_scores + attention_mask
        else:
            print('EROOR')
            exit()

        # 注意力分数的连接+softmax+dropout [batch_size, num_attention_heads, seq_length, seq_length + seq_length_other]
        attention_scores = torch.cat([attention_scores, other_attention_scores], dim=-1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # value的连接 [batch_size, num_attention_heads, seq_length + seq_length_other, attention_head_size]
        value_layer = torch.cat([value_layer, other_value_layer], dim=-2)

        # 最终的输出 [batch_size, seq_length, hidden_size]
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch_size, num_attention_heads, seq_length, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_length, num_attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  #
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    """
    Bert  Multi-Head Attention后的Add&Norm
    """
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        LayerNorm(dropout(FC(hidden_states)) + input_tensor)
        :param hidden_states: [batch_size, seq_length, hidden_size]
        :param input_tensor: [batch_size, seq_length, hidden_size]
        :return:
            [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """
    Bert  Multi-Head Attention + Add&Norm，组件：BertSelfAttention + BertSelfOutput
    """
    def __init__(self, seq_type, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(seq_type, config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, input_tensor_other, attention_mask):
        """

        :param input_tensor: [batch_size, seq_length, hidden_size]
        :param input_tensor_other: [batch_size, seq_length_other, hidden_size]
        :param attention_mask: [batch_size, 1, 1, batch_max_len_sentence] can broadcast to[batch_size, num_heads, from_seq_length, batch_max_len_sentence]
        :return:
             context_layer [batch_size, seq_length, hidden_size]
        """
        self_output = self.self(input_tensor, input_tensor_other, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    """
    Bert  FeedForward
    """
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """
        activate(FC(hidden_states))
        :param hidden_states: [batch_size, seq_length, hidden_size]
        :return:
            [batch_size, seq_length, config.intermediate_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    Bert  FeedForward后的Add&Norm
    """
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        LayerNorm(dropout(FC(hidden_states)) + input_tensor)
        BertIntermediate + BertOutput 即 FeedForward + Add&Norm
        :param hidden_states: [batch_size, seq_length, config.intermediate_size]
        :param input_tensor: [batch_size, seq_length, hidden_size]
        :return:
            [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    自定义的Bert层
    """
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention_text = BertAttention('TXT', config)
        self.intermediate_text = BertIntermediate(config)
        self.output_text = BertOutput(config)

        self.attention_visual = BertAttention('VIS', config)
        self.intermediate_visual = BertIntermediate(config)
        self.output_visual = BertOutput(config)

    def forward(self, hidden_states, hidden_states_other, attention_mask):
        """

        :param hidden_states: [batch_size, seq_length, hidden_size]
        :param hidden_states_other: [batch_size, seq_length_other, hidden_size]
        :param attention_mask: [batch_size, 1, 1, batch_max_len_sentence] can broadcast to[batch_size, num_heads, from_seq_length, batch_max_len_sentence]
        :return:
             layer_output [batch_size, seq_length, hidden_size]
             layer_output_other [batch_size, seq_length_other, hidden_size]
        """
        # MultiHeadAttention + Add&Norm
        attention_output = self.attention_text(hidden_states, hidden_states_other, attention_mask)
        attention_output_other = self.attention_visual(hidden_states_other, hidden_states, attention_mask)

        # FeedForward + Add&Norm
        intermediate_output = self.intermediate_text(attention_output)
        layer_output = self.output_text(intermediate_output, attention_output)

        intermediate_output_other = self.intermediate_visual(attention_output_other)
        layer_output_other = self.output_visual(intermediate_output_other, attention_output_other)

        return layer_output, layer_output_other


class BertEncoder(nn.Module):
    """
    自定义的Bert编码器
    """
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states_other, attention_mask):
        """

        :param hidden_states: [batch_size, seq_length, hidden_size]
        :param hidden_states_other: [batch_size, seq_length_other, hidden_size]
        :param attention_mask: [batch_size, 1, 1, batch_max_len_sentence] can broadcast to[batch_size, num_heads, from_seq_length, batch_max_len_sentence]
        :return:
            输出all_encoder_layers [1, 2] 2: [batch_size, seq_length, hidden_size] [batch_size, seq_length_other, hidden_size]

        """
        for layer_module in self.layer:
            hidden_states, hidden_states_other = layer_module(hidden_states, hidden_states_other, attention_mask)
        return [[hidden_states, hidden_states_other]]
