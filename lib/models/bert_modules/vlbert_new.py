# In the original release of the code, we inadvertently used the time-consuming masked_select and index_select operations inside loops. We fixed this in an updated version of vlbert_new.py by using masks to cover invalid positions in the 2D temporal map, but did not release new checkpoints for time reasons. Theoretically, the fix does not affect model performance, only FLOPs (see vlbert_new.py for details).

import copy
import torch
import torch.nn as nn
from lib.models.bert_modules.visual_linguistic_bert import VisualLinguisticBert
from lib.models.bert_modules.modeling import BertLayerNorm
from torch.nn import functional as F


class VLDCNP(nn.Module):
    def __init__(self, iou_mask_map, config):
        """

        :param iou_mask_map: [num_clips + 1, num_clips + 1]
        :param config:
        """
        super(VLDCNP, self).__init__()
        self.register_buffer('iou_mask_map', iou_mask_map)

        self.vldcnp_scale = config.vldcnp_scale

        self.proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm = BertLayerNorm(3 * config.hidden_size, eps=1e-12)

    def forward(self, vis_fea, txt_fea):
        """

        :param vis_fea: [b, num_clips + 1, num_clips + 1, 3 * hidden_size]
        :param txt_fea:  [b, batch_max_len_sentence, hidden_size]
        :return:
            [b, num_clips + 1, num_clips + 1, 3 * hidden_size]
        """

        tb, tT, _, thidden_size = vis_fea.shape

        # [b, num_moment, 3 * hidden_size]
        vis_fea = torch.masked_select(vis_fea, self.iou_mask_map.bool().unsqueeze(dim=0).unsqueeze(dim=3)).reshape(tb, -1, thidden_size)

        # 块1
        # [b, num_moment, 3 * hidden_size]
        v = vis_fea
        # [b, batch_max_len_sentence, 3 * hidden_size]
        q = F.relu(self.proj(txt_fea))
        a = v @ q.transpose(1, 2)
        sv = F.softmax(a, -1) @ q

        # 输出
        new_vis_fea = self.layerNorm(self.vldcnp_scale * self.dropout(sv) + vis_fea)

        indexes = self.iou_mask_map.flatten().nonzero().squeeze()
        out_tensor = torch.zeros(tb, tT * tT, thidden_size).to(device=vis_fea.device)
        out_tensor[:, indexes, :] = new_vis_fea
        return out_tensor.reshape(tb, tT, tT, thidden_size)


class DCGCN(nn.Module):
    """
    Double Connection Graph Convolution Neural Network
    """

    def __init__(self, iou_mask_map, config):
        """

        :param iou_mask_map: [num_clips + 1, num_clips + 1]
        :param config:
        """
        super(DCGCN, self).__init__()
        self.register_buffer('iou_mask_map', iou_mask_map)

        graph_hidden_size = 3 * config.hidden_size
        graph_layer_num = config.graph_layer_num
        proj_row = nn.Linear(graph_hidden_size, graph_hidden_size)
        proj_col = nn.Linear(graph_hidden_size, graph_hidden_size)
        layerNorm = BertLayerNorm(graph_hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.proj_rows = nn.ModuleList([copy.deepcopy(proj_row) for _ in range(graph_layer_num)])
        self.proj_cols = nn.ModuleList([copy.deepcopy(proj_col) for _ in range(graph_layer_num)])
        self.layerNorms = nn.ModuleList([copy.deepcopy(layerNorm) for _ in range(graph_layer_num)])

    def forward(self, input_tensor):
        """
        N_i_j = LayerNorm(dropout(relu(W @ N_0_j + W @ N_i_-1 + b)) + N_i_j)
        :param input_tensor: [b, num_clips + 1, num_clips + 1, 3 * hidden_size]
        :return:
            [b, num_clips + 1, num_clips + 1, 3 * hidden_size]
        """
        tb, tT, _, thidden_size = input_tensor.shape

        # [b, num_clips + 1, num_clips + 1, 3 * hidden_size]
        for proj_row, proj_col, layerNorm in zip(self.proj_rows, self.proj_cols, self.layerNorms):
            first_row_nodes = input_tensor[:, 0, :, :]
            last_col_nodes = input_tensor[:, :, -1, :]
            new_nodes = torch.relu(proj_row(first_row_nodes).unsqueeze(1).expand(tb, tT, tT, thidden_size)
                                   + proj_col(last_col_nodes).unsqueeze(2).expand(tb, tT, tT, thidden_size))
            input_tensor = layerNorm(self.dropout(new_nodes) + input_tensor) * self.iou_mask_map.unsqueeze(dim=0).unsqueeze(dim=3)

        return input_tensor


class TLocVLBERT(nn.Module):

    def __init__(self, dataset, config):
        super(TLocVLBERT, self).__init__()

        self.config = config

        # 生成2D时间图mask shape[num_clips+1, num_clips+1]
        if dataset == "ActivityNet":
            iou_mask_map = torch.zeros(33, 33).float()
            for i in range(0, 32, 1):
                iou_mask_map[i, i + 1:min(i + 17, 33)] = 1.
            for i in range(0, 32 - 16, 2):
                iou_mask_map[i, range(18 + i, 33, 2)] = 1.
        elif dataset == "TACoS":
            iou_mask_map = torch.zeros(129, 129).float()
            for i in range(0, 128, 1):
                iou_mask_map[i, 1 + i:min(i + 17, 129)] = 1.
            for i in range(0, 128 - 16, 2):
                iou_mask_map[i, range(18 + i, min(33 + i, 129), 2)] = 1.
            for i in range(0, 128 - 32, 4):
                iou_mask_map[i, range(36 + i, min(65 + i, 129), 4)] = 1.
            for i in range(0, 128 - 64, 8):
                iou_mask_map[i, range(72 + i, 129, 8)] = 1.
        elif dataset == "Charades":
            iou_mask_map = torch.zeros(65, 65).float()
            for i in range(0, 64, 1):
                iou_mask_map[i, 1 + i:min(i + 17, 65)] = 1.
            for i in range(0, 64 - 16, 2):
                iou_mask_map[i, range(18 + i, min(33 + i, 65), 2)] = 1.
            for i in range(0, 64 - 32, 4):
                iou_mask_map[i, range(36 + i, min(65 + i, 65), 4)] = 1.
            for i in range(0, 64 - 64, 8):
                iou_mask_map[i, range(72 + i, 65, 8)] = 1.
        else:
            print('DATASET ERROR')
            exit()
        self.register_buffer('iou_mask_map', iou_mask_map)

        self.vlbert = VisualLinguisticBert(dataset, config)

        dim = config.hidden_size

        if config.CLASSIFIER_TYPE == "2fc":
            # word prediction: hidden_size -> vocab_size
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, config.vocab_size)
            )
            # generate stage-specific feature: hidden_size -> 3 * hidden_size
            self.final_mlp_2 = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, dim * 3),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
            )
            # generate align score and regression score from moment candidate feature: 3 * hidden_size -> 3
            self.final_mlp_3 = torch.nn.Sequential(
                torch.nn.Linear(3 * dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 3)
            )
            # generate backfore score -> 1
            self.final_mlp_backfore = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.CLASSIFIER_TYPE))

        # 对final_mlp、final_mlp_2、final_mlp_3中的linear层权重按照xavier_uniform_初始化
        self.init_weight()

        self.vldcnp = VLDCNP(iou_mask_map, config)
        self.dcgcn = DCGCN(iou_mask_map, config)

    def init_weight(self):
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_2.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_3.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, text_input_feats, text_mask, word_mask, object_visual_feats):
        """

        :param text_input_feats: 单词特征 shape[b, batch_max_len_sentence, text_fea_len] tensor float32 经0-pad_sequence
        :param text_mask: 文本掩码 shape[b, batch_max_len_sentence] ones||zeros tensor float32 经0-pad_sequence
        :param word_mask: 单词掩码 shape[b, batch_max_len_sentence] tensor float32 1.-mask 0.-unmask 15%的mask率 经0-pad_sequence
        :param object_visual_feats: 视觉特征 shape[b, num_clips + 1, vis_fea_len] tensor float32
        :return:
            logits_text: [b, batch_max_len_sentence, vocab_size] predict mask word(包含pad出单词的预测，但后面仅对遮盖单词计算损失)
            logits_iou: [b, 3, num_clips + 1, num_clips + 1] 任意开始-任意终止时间点的 回归分数+对齐分数
            iou_mask_map.clone().detach(): [num_clips + 1, num_clips + 1] 2D时间图mask
            logits_backfore: 前背景概率值 [b, num_clips + 1]
        """
        # Visual Linguistic BERT
        # -> text_output[b, batch_max_len_sentence, hidden_size] visual_output[b, num_clips + 1, hidden_size]
        hidden_states_text_raw, hidden_states_object_raw, text_embeddings = self.vlbert(text_input_feats,
                                                               text_mask,
                                                               word_mask,
                                                               object_visual_feats)

        # predict mask word: [b, batch_max_len_sentence, hidden_size] -> [b, batch_max_len_sentence, vocab_size]
        logits_text = self.final_mlp(hidden_states_text_raw)

        # generate stage-specific feature [b, num_clips + 1, hidden_size] -> 3 * [b, num_clips + 1, hidden_size]
        hidden_states_object = self.final_mlp_2(hidden_states_object_raw)
        hidden_s, hidden_e, hidden_c = torch.split(hidden_states_object, self.config.hidden_size, dim=-1)

        # c_point [b, num_clips + 1, num_clips + 1, hidden_size] 任意开始-任意终止时间点的 中间特征
        T = hidden_states_object.size(1)
        s_idx = torch.arange(T, device=hidden_states_object.device)
        e_idx = torch.arange(T, device=hidden_states_object.device)
        c_point = hidden_c[:, (0.5 * (s_idx[:, None] + e_idx[None, :])).long().flatten(), :].view(hidden_c.size(0), T, T, hidden_c.size(-1))
        # s_c_e_point [b, num_clips + 1, num_clips + 1, 3*hidden_size] 任意开始-任意终止时间点的 联合特征(start-middle-end)
        s_c_e_points = torch.cat((hidden_s[:, :, None, :].repeat(1, 1, T, 1), c_point, hidden_e[:, None, :, :].repeat(1, T, 1, 1)), -1)

        # for target
        s_c_e_points = self.vldcnp(s_c_e_points, text_embeddings)

        # optional DCGCN
        s_c_e_points = self.dcgcn(s_c_e_points)

        # logits_iou [b, 3, num_clips + 1, num_clips + 1] 任意开始-任意终止时间点的 回归分数+对齐分数
        logits_iou = self.final_mlp_3(s_c_e_points).permute(0, 3, 1, 2).contiguous()

        logits_backfore = self.final_mlp_backfore(hidden_states_object_raw).squeeze(dim=2)

        return logits_text, logits_iou, self.iou_mask_map.clone().detach(), logits_backfore
