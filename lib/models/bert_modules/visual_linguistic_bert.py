from __future__ import division
import math
import numpy as np
import torch
import torch.nn as nn
from lib.models.bert_modules.modeling import BertLayerNorm, BertEncoder


class PositionalEncoding(nn.Module):
    """
    添加位置编码

    __init__
        :param d_hid: d_model
        :param n_position: 可以处理的最大seq_len，default(116)

    forward
        :param x: shape[batch_size, seq_len, d_model]
        :return:
            添加位置编码后的x shape[batch_size, seq_len, d_model]
    """
    def __init__(self, d_hid, n_position=116):
        """

        :param d_hid: d_model
        :param n_position: 可以处理的最大seq_len，default(116)
        """
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i / n_position) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2] * 2 * math.pi)  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2] * 2 * math.pi)  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """

        :param x: shape[batch_size, seq_len, d_model]
        :return:
            添加位置编码后的x shape[batch_size, seq_len, d_model]
        """
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        self.config = config
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """
        Initialize the weights.

        if module 属于 Linear或Embedding:
            weight 填充为 正态分布(均值0, 标准差config.initializer_range)

        if module 属于 Linear且有bias:
            bias   填充为0

        if module 属于 BertLayerNorm:
            weight 填充为 1
            bias   填充为 0
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented


class VisualLinguisticBert(BaseModel):
    def __init__(self, dataset, config):
        super(VisualLinguisticBert, self).__init__(config)

        self.config = config

        # embedding函数内使用
        self.mask_embeddings = nn.Embedding(1, config.hidden_size)
        self.word_mapping = nn.Linear(300, config.hidden_size)
        self.text_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.text_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.visual_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.visual_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.visual_1x1_object = None
        if config.visual_size != config.hidden_size:
            self.visual_1x1_object = nn.Linear(config.visual_size, config.hidden_size)
        if config.visual_ln:
            self.visual_ln_object = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.vis_sep_enc = nn.GRU(config.hidden_size, config.hidden_size // 2, num_layers=config.vis_enc_layers,
                                  batch_first=True, dropout=config.hidden_dropout_prob, bidirectional=True)
        self.txt_sep_enc = nn.GRU(config.hidden_size, config.hidden_size // 2, num_layers=config.txt_enc_layers,
                                  batch_first=True, dropout=config.hidden_dropout_prob, bidirectional=True)

        self.encoder = BertEncoder(config)

        # init weights
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_ln_object.weight.data.fill_(self.config.visual_scale_object_init)


    def forward(self,
                text_input_feats,
                text_mask,
                word_mask,
                object_visual_embeddings):
        """

        :param text_input_feats: 单词特征 shape[b, batch_max_len_sentence, text_fea_len] tensor float32 经0-pad_sequence
        :param text_mask: 文本掩码 shape[b, batch_max_len_sentence] ones||zeros tensor float32 经0-pad_sequence
        :param word_mask: 单词掩码 shape[b, batch_max_len_sentence] tensor float32 1.-mask 0.-unmask 15%的mask率 经0-pad_sequence
        :param object_visual_embeddings: 视觉特征 shape[b, num_clips + 1, vis_fea_len] tensor float32
        :return:
            text_output[b, batch_max_len_sentence, hidden_size]
            visual_output[b, num_clips + 1, hidden_size]
            text_embeddings: [b, batch_max_len_sentence, hidden_size]
        """

        # text_embeddings: [b, batch_max_len_sentence, hidden_size] visual_embeddings: [b, num_clips + 1, hidden_size]
        text_embeddings, visual_embeddings = self.embedding(text_input_feats, text_mask, word_mask, object_visual_embeddings)

        # shape[batch_size, 1, 1, batch_max_len_sentence]  can broadcast to[batch_size, num_heads, from_seq_length, batch_max_len_sentence]
        extended_attention_mask = text_mask.unsqueeze(1).unsqueeze(2)

        # 注意力分数遮盖 将其添加到softmax之前的原始分数上(不遮盖位置+0，遮盖位置-1000000.0)
        # softmax后原1位置不变、原0位置为0
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0

        # for data
        self.txt_sep_enc.flatten_parameters()
        self.vis_sep_enc.flatten_parameters()
        text_embeddings = self.txt_sep_enc(text_embeddings)[0]
        visual_embeddings = self.vis_sep_enc(visual_embeddings)[0]

        # for content
        encoded_layers = self.encoder(text_embeddings, visual_embeddings, extended_attention_mask)

        encoded_layers = encoded_layers[-1]
        return encoded_layers[0], encoded_layers[1], text_embeddings

    def embedding(self,
                  text_input_feats,
                  text_mask,
                  word_mask,
                  object_visual_embeddings):
        """
        文本特征维度 -> hidden_size
        (训练阶段)遮盖15%单词的特征(使用一个可学习的Embedding替代)
        视觉特征维度 -> hidden_size BertLayerNorm
        视觉信息、文本信息分别BertLayerNorm、dropout
        :param text_input_feats: 单词特征 shape[b, batch_max_len_sentence, text_fea_len] tensor float32 经0-pad_sequence
        :param text_mask: 文本掩码 shape[b, batch_max_len_sentence] ones||zeros tensor float32 经0-pad_sequence
        :param word_mask: 单词掩码 shape[b, batch_max_len_sentence] tensor float32 1.-mask 0.-unmask 15%的mask率 经0-pad_sequence
        :param object_visual_embeddings: 视觉特征 shape[b, num_clips + 1, vis_fea_len] tensor float32
        :return:
            text_embeddings: [b, batch_max_len_sentence, hidden_size]
            visual_embeddings: [b, num_clips + 1, hidden_size]
        """
        # [b, batch_max_len_sentence, text_fea_len] -> [b, batch_max_len_sentence, hidden_size]
        text_linguistic_embedding = self.word_mapping(text_input_feats)

        # 训练阶段，遮盖单词
        if self.training:
            _zero_id = torch.zeros(text_linguistic_embedding.shape[:2], dtype=torch.long,
                                   device=text_linguistic_embedding.device)
            text_linguistic_embedding[word_mask > 0] = self.mask_embeddings(_zero_id)[word_mask > 0]

        # [b, num_clips + 1, vis_fea_len] -> [b, num_clips + 1, hidden_size] BertLayerNorm
        if self.visual_1x1_object is not None:
            object_visual_embeddings = self.visual_1x1_object(object_visual_embeddings)
        if self.config.visual_ln:
            object_visual_embeddings = self.visual_ln_object(object_visual_embeddings)

        visual_embeddings, text_embeddings = object_visual_embeddings, text_linguistic_embedding

        # 对视觉信息 LayerNorm dropout
        text_embeddings = self.text_embedding_LayerNorm(text_embeddings)
        text_embeddings = self.text_embedding_dropout(text_embeddings)

        # 对文本信息 LayerNorm dropout
        visual_embeddings = self.visual_embedding_LayerNorm(visual_embeddings)
        visual_embeddings = self.visual_embedding_dropout(visual_embeddings)

        return text_embeddings, visual_embeddings
