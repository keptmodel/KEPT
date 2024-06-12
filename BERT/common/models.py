import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
import common.parse.DFG 

class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class RelationClassifyHeader(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)

        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
class RelationClassifyCLS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, code_hidden, text_hidden):
        code_dim = code_hidden.dim()
        text_dim = text_hidden.dim()
        if code_dim==3:
            pool_code_hidden = code_hidden[:, 0, :]
        else:
            pool_code_hidden = code_hidden[:,0,0,:]
        if text_dim==3:
            pool_text_hidden = text_hidden[:, 0, :]
        else:
            pool_text_hidden = text_hidden[:,0,0,:]
            
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class Rapt(PreTrainedModel):
    def __init__(self, config, code_bert):
        super().__init__(config)
        cbert_model = code_bert

        self.ctokneizer = AutoTokenizer.from_pretrained(cbert_model, local_files_only=True)
        self.cbert = AutoModel.from_pretrained(cbert_model, local_files_only=True)

        self.ntokenizer = self.ctokneizer
        self.nbert = self.cbert

        self.cls = RelationClassifyCLS(config)

    def forward(
            self,
            code_ids=None,
            code_pos=None,
            code_attention_mask=None,
            text_ids=None,
            text_pos=None,
            text_attention_mask=None,
            relation_label=None):
        c_hidden = self.cbert(code_ids, position_ids=code_pos, attention_mask=code_attention_mask)[0]
        n_hidden = self.nbert(text_ids, position_ids=text_pos, attention_mask=text_attention_mask)[0]
        #code_mask = code_attention_mask[:,0,:]
        #c_hidden = (c_hidden*code_mask.ne(0)[:,:,None]).sum(1)/code_mask.ne(0).sum(-1)[:,None]
        #c_hidden = c_hidden.unsqueeze(1)
        
        #text_mask = text_attention_mask[:,0,:]
        #n_hidden = (n_hidden*text_mask.ne(0)[:,:,None]).sum(1)/text_mask.ne(0).sum(-1)[:,None]
        #n_hidden = n_hidden.unsqueeze(1)
        
        logits = self.cls(code_hidden=c_hidden, text_hidden=n_hidden)
        output_dict = {"logits": logits}
        if relation_label is not None:
            loss_fct = CrossEntropyLoss()
            rel_loss = loss_fct(logits.view(-1, 2), relation_label.view(-1))
            output_dict['loss'] = rel_loss
        return output_dict  # (rel_loss), rel_score

    def get_sim_score(self, text_hidden, code_hidden):
        logits = self.cls(text_hidden=text_hidden, code_hidden=code_hidden)
        sim_scores = torch.softmax(logits, 1).data.tolist()
        return [x[1] for x in sim_scores]

    def get_nl_tokenizer(self):
        return self.ntokenizer

    def get_pl_tokenizer(self):
        return self.ctokneizer

    def create_nl_embd(self, input_ids,pos, attention_mask):
        n_hidden = self.nbert(input_ids, position_ids=pos, attention_mask=attention_mask)[0]
        #text_mask = attention_mask[:,0,:]
        #n_hidden = (n_hidden*text_mask.ne(0)[:,:,None]).sum(1)/text_mask.ne(0).sum(-1)[:,None]
        #n_hidden = n_hidden.unsqueeze(1)
        return n_hidden

    def create_pl_embd(self, input_ids,pos, attention_mask):
        c_hidden = self.cbert(input_ids, position_ids=pos, attention_mask=attention_mask)[0]
        #code_mask = attention_mask[:,0,:]
        #c_hidden = (c_hidden*code_mask.ne(0)[:,:,None]).sum(1)/code_mask.ne(0).sum(-1)[:,None]
        #c_hidden = c_hidden.unsqueeze(1)
        return c_hidden
    
    def get_nl_sub_model(self):
        return self.nbert

    def get_pl_sub_model(self):
        return self.cbert
