import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.qa_helper import *
    

class LangModule(nn.Module):
    def __init__(self, num_object_class, use_lang_classifier=True, use_bidir=False, num_layers=1,
        emb_size=300, hidden_size=256, pdrop=0.1, word_pdrop=0.1, 
        bert_model_name=None, freeze_bert=False, finetune_bert_last_layer=False):
        super().__init__() 

        self.num_object_class = num_object_class
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.num_layers = num_layers
        self.bert_model_name = bert_model_name
        self.use_bert_model = bert_model_name is not None

        #根据选择的配置对 BERT 模型进行初始化和微调。可能的配置包括冻结整个 BERT 模型、仅微调最后一层
            #或 transformer layer。这种微调操作允许在特定任务上使用 BERT 模型并进行进一步的训练。
        if self.use_bert_model:
            from transformers import AutoModel 
            self.bert_model = AutoModel.from_pretrained(bert_model_name)
            #两个选项决定是否冻结整个 BERT 模型或者仅微调最后一层。确保两者不同时为true
            assert not (freeze_bert and finetune_bert_last_layer)
            if freeze_bert:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            elif finetune_bert_last_layer:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
                if hasattr(self.bert_model, 'encoder'):
                    for param in self.bert_model.encoder.layer[-1].parameters():
                        param.requires_grad = True
                else: # distill-bert
                    for param in self.bert_model.transformer.layer[-1].parameters():
                        param.requires_grad = True                    

       
        #lstm处理文本和时间序列
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,#指定输入和输出张量的第一个维度是否是批量大小
            num_layers=num_layers,
            bidirectional=use_bidir,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.word_drop = nn.Dropout(pdrop)

        lang_size = hidden_size * 2 if use_bidir else hidden_size

        #
        # Language classifier
        #   num_object_class -> 18
        if use_lang_classifier:
            #表示包含dropout和线性层两层
            self.lang_cls = nn.Sequential(
                nn.Dropout(p=pdrop),
                nn.Linear(lang_size, num_object_class),
                #nn.Dropout()
            )

    def make_mask(self, feature):
        """
        return a mask that is True for zero values and False for other values.
        """
        #判断feature沿最后一个维度的绝对值之和是否为0
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0) #.unsqueeze(-1) #.unsqueeze(2)        


    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        if hasattr(self, 'bert_model'):
            word_embs = self.bert_model(**data_dict["lang_feat"])
            word_embs = word_embs.last_hidden_state # batch_size, MAX_TEXT_LEN (32), bert_embed_size
        else:
            word_embs = data_dict["lang_feat"] # batch_size, MAX_TEXT_LEN (32), glove_size

        # dropout word embeddings
        word_embs = self.word_drop(word_embs)
        lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"].cpu(), batch_first=True, enforce_sorted=False)

        # encode description
        packed_output, (lang_last, _) = self.lstm(lang_feat)
        lang_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        data_dict["lang_out"] = lang_output # batch_size, num_words(max_question_length), hidden_size * num_dir

        # lang_last: (num_layers * num_directions, batch_size, hidden_size)
        #将多层循环神经网络最后一层的隐藏状态处理成一个形状为 (batch_size, hidden_size * num_directions) 的张量
        _, batch_size, hidden_size = lang_last.size()
        lang_last = lang_last.view(self.num_layers, -1, batch_size, hidden_size) 
        # lang_last: num_directions, batch_size, hidden_size
        lang_last = lang_last[-1]
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir

        # store the encoded language features
        #将处理过的编码语言特征存储在数据字典中，并生成相应的语言掩码。语言掩码的目的是标记输入序列中的哪些位置是有效的，哪些是填充的或者无效的
        data_dict["lang_emb"] = lang_last # batch_size, hidden_size * num_dir
        if self.use_bert_model:
            data_dict["lang_mask"] = ~data_dict["lang_feat"]["attention_mask"][:,:lang_output.shape[1]].bool() # batch_size, num_words (max_question_length)
        else:
            data_dict["lang_mask"] = self.make_mask(lang_output) # batch_size, num_words (max_question_length)

        # classify
        #nlp中常见于文本分类或情感分析
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])
        return data_dict
