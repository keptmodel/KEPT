import os

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, PreTrainedModel,RobertaModel
from common.parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from common.parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
import torch.nn.functional as F


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
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dense(concated_hidden)
        x = torch.tanh(x)
        x = self.output_layer(x)
        return x
parsers={}
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}
current_work_dir = os.path.dirname(__file__)
for lang in dfg_function:
    LANGUAGE = Language(os.path.join(current_work_dir,'parser/my-languages.so'), lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser,dfg_function[lang]]
    parsers[lang]= parser
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"
    try:
        code = code[:10000]
        tree = parser[0].parse(bytes(code,'utf8'))
        root_node = tree.root_node
        tokens_index=tree_to_token_index(root_node)
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)
        try:
            DFG,_=parser[1](root_node,index_to_code,{})
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except Exception as e:
        dfg=[]
        code_tokens=None
    return code_tokens,dfg

def convert_code_to_features(code,code_limit,dfg_limit,tokenizer):
    #code
    parser=parsers["java"]
    #extract data flow
    code_tokens,dfg=extract_dataflow(code,parser,"java")
    if code_tokens is None:
        return tokenizer.encode_plus(code, max_length=dfg_limit+code_limit, truncation=True,
                                        padding='max_length', return_attention_mask=True,
                                        return_token_type_ids=False)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))
    code_tokens=[y for x in code_tokens for y in x]
    #truncating
    code_tokens=code_tokens[:code_limit+dfg_limit-2-min(len(dfg),dfg_limit)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:code_limit+dfg_limit-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=code_limit+dfg_limit-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
    #nl
    return code_ids,position_idx,dfg_to_code,dfg_to_dfg

F_ATTEN_MASK = "attention_mask"
F_INPUT_ID = "input_ids"
F_POS = "pos"
class FakeTokenizer:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer

    def encoder_plus(self,pl,code_limit=256,dfg_limit=64):
        code_ids,position_idx,dfg_to_code,dfg_to_dfg = convert_code_to_features(pl,code_limit,dfg_limit,self.tokenizer)
        attn_mask = np.zeros((code_limit+dfg_limit,
                              code_limit+dfg_limit), dtype=np.bool_)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in position_idx])
        max_length = sum([i != 1 for i in position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(position_idx):
                    attn_mask[idx + node_index, a + node_index] = True
        return {F_ATTEN_MASK:attn_mask,F_INPUT_ID:code_ids,F_POS:position_idx}


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
        if code_dim == 3:
            pool_code_hidden = code_hidden[:, 0, :]
        elif code_dim==4:
            pool_code_hidden = code_hidden[:, 0, 0, :]
        else:
            pool_code_hidden = code_hidden
        if text_dim == 3:
            pool_text_hidden = text_hidden[:, 0, :]
        elif text_dim==4:
            pool_text_hidden = text_hidden[:, 0, 0, :]
        else:
            pool_text_hidden = text_hidden

        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class GCBert(PreTrainedModel):
    def __init__(self, config, gcbert):
        super().__init__(config)
        gcbert_model = gcbert

        self.ctokneizer = AutoTokenizer.from_pretrained(gcbert_model,local_files_only = True)
        self.ntokenizer = self.ctokneizer

        self.ctokneizer = FakeTokenizer(self.ctokneizer)

        self.cbert = RobertaModel.from_pretrained(gcbert_model,local_files_only = True)
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
        
        nodes_mask=code_pos.eq(0)
        token_mask=code_pos.ge(2)        
        inputs_embeddings=self.cbert.embeddings.word_embeddings(code_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&code_attention_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        c_hidden = self.cbert(inputs_embeds=inputs_embeddings,attention_mask=code_attention_mask,position_ids=code_pos)[0]
        n_hidden = self.nbert(text_ids, position_ids=text_pos, attention_mask=text_attention_mask)[0]
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

    def create_nl_embd(self, input_ids, pos, attention_mask):
        n_hidden = self.nbert(input_ids, position_ids=pos, attention_mask=attention_mask)[0]
        # text_mask = attention_mask[:,0,:]
        # n_hidden = (n_hidden*text_mask.ne(0)[:,:,None]).sum(1)/text_mask.ne(0).sum(-1)[:,None]
        # n_hidden = n_hidden.unsqueeze(1)
        return n_hidden

    def create_pl_embd(self, input_ids, pos, attention_mask):
        c_hidden = self.cbert(input_ids, position_ids=pos, attention_mask=attention_mask)[0]
        # code_mask = attention_mask[:,0,:]
        # c_hidden = (c_hidden*code_mask.ne(0)[:,:,None]).sum(1)/code_mask.ne(0).sum(-1)[:,None]
        # c_hidden = c_hidden.unsqueeze(1)
        return c_hidden

    def get_nl_sub_model(self):
        return self.nbert

    def get_pl_sub_model(self):
        return self.cbert




