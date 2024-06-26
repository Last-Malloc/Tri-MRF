from torch import nn
import lib.models.bert_modules as bert_modules
import lib.models.frame_modules as frame_modules
from lib.core.config import config


class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()
        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.bert_layer = getattr(bert_modules, config.TAN.VLBERT_MODULE.NAME)(config.DATASET.NAME, config.TAN.VLBERT_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, word_mask, visual_input):
        """

        :param textual_input: 单词特征 shape[b, batch_max_len_sentence, text_fea_len] tensor float32 经0-pad_sequence
        :param textual_mask: 文本掩码 shape[b, batch_max_len_sentence] ones||zeros tensor float32 经0-pad_sequence
        :param word_mask: 单词掩码 shape[b, batch_max_len_sentence] tensor float32 1.-mask 0.-unmask 15%的mask率 经0-pad_sequence
        :param visual_input: 视觉特征 shape[b, num_sample_clips, vis_fea_len] tensor float32
        :return:
            logits_text: [b, batch_max_len_sentence, vocab_size] predict mask word(包含pad出单词的预测，但后面仅对遮盖单词计算损失)
            logits_iou: [b, 3, num_clips + 1, num_clips + 1] 任意开始-任意终止时间点的 回归分数+对齐分数
            iou_mask_map: [num_clips + 1, num_clips + 1] 2D时间图mask
            logits_backfore: 前背景概率值 [b, num_clips + 1]
        """
        # conv1d shape[b, num_sample_clips, vis_fea_len] -> [b, num_clips + 1, vis_fea_len]
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        vis_h = vis_h.transpose(1, 2)

        # logits_text: [b, batch_max_len_sentence, vocab_size] predict mask word(包含pad出单词的预测，但后面仅对遮盖单词计算损失)
        # logits_iou: [b, 3, num_clips + 1, num_clips + 1] 任意开始-任意终止时间点的 回归分数+对齐分数
        # iou_mask_map: [num_clips + 1, num_clips + 1] 2D时间图mask 1.-select 0.-unselect
        # logits_backfore: 前背景概率值 [b, num_clips + 1]
        logits_text, logits_iou, iou_mask_map, logits_backfore = self.bert_layer(textual_input, textual_mask, word_mask, vis_h)

        return logits_text, logits_iou, iou_mask_map, logits_backfore
