import functools

import datrie
import numpy as np
import pandas as pd
from pygments import lex
from pygments.lexers import JavaLexer
from tqdm import tqdm
from collections import defaultdict
import json
import os

def caculate_base_sent_tree(split_tokens, max_length, tokenizer):
    # insert relations
    pos_idx_tree = []
    abs_idx_tree = []
    pos_idx = -1
    abs_idx = -1
    abs_idx_src = []
    for tokens_p, rels in split_tokens:
        token_pos_idx = [pos_idx + i for i in range(1, len(tokens_p) + 1)]
        token_abs_idx = [abs_idx + i for i in range(1, len(tokens_p) + 1)]
        abs_idx = token_abs_idx[-1]

        entities_pos_idx = []
        entities_abs_idx = []
        for rel in rels:
            ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(rel) + 1)]
            entities_pos_idx.append(ent_pos_idx)
            ent_abs_idx = [abs_idx + i for i in range(1, len(rel) + 1)]
            abs_idx = ent_abs_idx[-1]
            entities_abs_idx.append(ent_abs_idx)

        pos_idx_tree.append((token_pos_idx, entities_pos_idx))
        pos_idx = token_pos_idx[-1]
        abs_idx_tree.append((token_abs_idx, entities_abs_idx))
        abs_idx_src += token_abs_idx

    # generate sentence with triplets
    know_sent = []
    pos = []
    seg = []
    for i in range(len(split_tokens)):
        add_split_tokens = split_tokens[i][0]
        know_sent += add_split_tokens
        seg += [0] * len(add_split_tokens)
        pos += [pos + 2 for pos in pos_idx_tree[i][0]]
        for j in range(len(split_tokens[i][1])):
            add_rel = list(split_tokens[i][1][j])
            know_sent += add_rel
            seg += [1] * len(add_rel)
            pos += [pos + 2 for pos in list(pos_idx_tree[i][1][j])]

    token_num = len(know_sent)

    # Calculate visible matrix
    visible_matrix = np.zeros((token_num, token_num),dtype=int)
    for item in abs_idx_tree:
        src_ids = item[0]
        for id in src_ids:
            visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
            visible_matrix[id, visible_abs_idx] = 1
        for ent in item[1]:
            for id in ent:
                visible_abs_idx = ent + src_ids
                visible_matrix[id, visible_abs_idx] = 1

    src_length = len(know_sent)
    if len(know_sent) < max_length:
        pad_num = max_length - src_length
        know_sent += [tokenizer.pad_token] * pad_num
        seg += [0] * pad_num
        pos += [1] * pad_num
        visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
        #add_line = visible_matrix[0]
        #for i in range(src_length,max_length):
        #    visible_matrix =  np.append(visible_matrix, [add_line], axis=0)
            

    sent_ids = tokenizer.convert_tokens_to_ids(know_sent)
    return {'input_ids': sent_ids, 'pos': pos, 'seg': seg, 'attention_mask': visible_matrix}


REL_PRIORITY_MAP = {
    'CALL': 10,
    'INHERITANCE': 2,
    'IMPLEMENT': 1,
    'HAS': 4,
    'INSTANCEOF': 3,
    'RETURN': 5,
    'HASPARAMETER': 6,
    'HASVARIABLE': 7
}
REL_ATTRIBUTE_MAP = {
    'CALL': [],
    'INHERITANCE': ['ONLYONE'],
    'IMPLEMENT': [],
    'HAS': ['ACCURACY'],
    'RETURN': ['ONLYONE'],
    'INSTANCEOF': ['ONLYONE'],
    'HASPARAMETER': ['ACCURACY'],
    'HASVARIABLE': ['ACCURACY']
}
REL_RENAME_MAP = {
    'CALL': 'invoke',
    'INHERITANCE': 'extend',
    'IMPLEMENT': 'implement',
    'HAS': 'has member',
    'INSTANCEOF': 'is instance of',
    'RETURN': 'return type',
    'HASPARAMETER': 'has parameter',
    'HASVARIABLE': 'has variable'
}


class Entity:
    def __init__(self, name, entity_id, identifier, property):
        self.name = name
        self.entity_id = entity_id
        self.identifier = identifier
        self.relations = []
        self.property = property

    def add_relation(self, relation, tail):
        self.relations.append((relation, tail))


class CodeKnowledgeGraph:
    def __init__(self, data_path: str, tokenizer,args,set_path=None,other_rel_map=None):
        self.args = args
        self.identifier_tree: datrie.Trie = datrie.Trie(ranges=[(chr(32), chr(127))])
        self.name_map = {}
        self.id_map = {}
        self.tokenizer = tokenizer
        self.statistics = {"rel_num":0,"data_num":0}
        self.__load_data(data_path,set_path)
        self.entity_flag = '<entity_flag_nvjre>'
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.entity_flag]})
        self.rel_count_map=defaultdict(int)
        self.rel_set = set()
        self.other_rel_map = other_rel_map
        self.record =defaultdict(list)

    def __load_data(self, data_path: str,set_path):
        if data_path=="":
            return
        entity_path = data_path + "code_entity.csv"
        relation_path = data_path + "code_relation.csv"
        entity_pd = pd.read_csv(entity_path)
        relation_path = pd.read_csv(relation_path)
        for index, row in tqdm(entity_pd.iterrows(), total=entity_pd.shape[0],
                               desc="Loading Code Knowledge Graph Entity",maxinterval = 3600,mininterval=self.args.tqdm_interval):
            if pd.isna(row['NAME']) or pd.isna(row["IDNAME"]):
                continue
            entity = Entity(row['NAME'], row['ID'], row["IDNAME"], row["PROPERTY"])
            if entity.name in self.name_map:
                self.name_map[entity.name].append(entity)
            else:
                self.name_map[entity.name] = [entity]
            self.id_map[entity.entity_id] = entity
            self.identifier_tree[entity.identifier] = entity

        for index, row in tqdm(relation_path.iterrows(), total=relation_path.shape[0],maxinterval = 3600,mininterval=self.args.tqdm_interval,
                               desc="Loading Code Knowledge Graph Relation"):
            head = self.id_map[row["HEAD"]] if row["HEAD"] in self.id_map else None
            tail = self.id_map[row["TAIL"]] if row["TAIL"] in self.id_map else None
            if head is None or tail is None:
                continue
            head.add_relation(row["PROPERTY"], tail)
    def load_set(self,set_path):
        if not os.path.exists(os.path.join(set_path,"rel_set")):
            return
        with open(os.path.join(set_path,"rel_set"),'r') as f:
            rel_set_list = json.load(f)
        self.rel_set = set(rel_set_list)       
    def __code_cut(self, code):
        tokens = lex(code, JavaLexer())
        return [token[1] for token in tokens]

    def __find_entitys(self, first_map, second_map, sentence):
        tokens = self.__code_cut(sentence)
        if tokens[-1]=='\n':
            tokens = tokens[:-1]
        tag_tokens = []
        token_set = set(tokens)

        entity_set = set()

        entity_list = []

        for index, token in enumerate(tokens):
            if token in entity_set:
                tag_tokens.append(token)
                continue
            if token in first_map.keys():
                tag_tokens.append(self.entity_flag)
                tag_tokens.append(token)
                tag_tokens.append(self.entity_flag)

                entity_set.add(token)
                entity_list.append((1, first_map[token]))
            elif token in second_map.keys():
                entitys = second_map[token]
                entity_set.add(token)

                if len(entitys) == 1:
                    entity = entitys[0]
                else:
                    tag_tokens.append(token)
                    continue

                tag_tokens.append(self.entity_flag)
                tag_tokens.append(token)
                tag_tokens.append(self.entity_flag)

                entity_list.append((2, entity))
            else:
                tag_tokens.append(token)
                continue
        return tag_tokens, entity_list

    def find_relation(self, entity_loc_list, tokens_limit,id):
        entity_set = set([value['entity'][1] for value in entity_loc_list])

        pri_rel_list = []
        for entity_loc in entity_loc_list:
            prior = entity_loc['entity'][0]
            entity = entity_loc['entity'][1]
            loc = entity_loc['location']
            for rel, tail in entity.relations:
                if self.args.code_kg_mode =="other":
                    if tail in entity_set:
                        continue
                    pri_rel_list.append((prior, loc, entity, rel, tail))
                elif self.args.code_kg_mode =="inner":
                    if tail in entity_set:
                        pri_rel_list.append((prior, loc, entity, rel, tail))
                    

        def rel_sort_func(rel_a, rel_b):
            prior_a, loc_a, entity_a, rel_a, tail_a = rel_a
            prior_b, loc_b, entity_b, rel_b, tail_b = rel_b
            if prior_a < prior_b:
                return -1
            if prior_a > prior_b:
                return 1
            if REL_PRIORITY_MAP[rel_a] < REL_PRIORITY_MAP[rel_b]:
                return -1
            if REL_PRIORITY_MAP[rel_a] > REL_PRIORITY_MAP[rel_b]:
                return 1
            return 0

        pri_rel_list.sort(key=functools.cmp_to_key(rel_sort_func))

        candidate_rel_list = []
        for relation in pri_rel_list:
            prior, loc, entity, rel, tail = relation
            
            sent_tree = REL_RENAME_MAP[rel] + ' ' + tail.name
            
            sent_tree = self.tokenizer.tokenize(sent_tree)
            if len(sent_tree) > tokens_limit:
                continue
            if entity.name+' '+REL_RENAME_MAP[rel]+' '+tail.name not in self.rel_set:
                if self.other_rel_map is not None and self.other_rel_map[entity.name+' '+REL_RENAME_MAP[rel]+' '+tail.name]<=1:
                    self.record[id].append(entity.name+'|'+REL_RENAME_MAP[rel]+'|'+tail.name)
                    continue
            tokens_limit -= len(sent_tree)
            self.statistics["rel_num"]+=1
            self.rel_count_map[entity.name+' '+REL_RENAME_MAP[rel]+' '+tail.name]+=1
            candidate_rel_list.append((loc, entity, sent_tree))

        candidate_rels_map = {}
        for candidate_rel in candidate_rel_list:
            loc, entity, sent_tree = candidate_rel
            if loc in candidate_rels_map:
                candidate_rels_map[loc].append(sent_tree)
            else:
                candidate_rels_map[loc] = [sent_tree]

        return candidate_rels_map

    def add_relation_encode(self, pl_txt: str, max_length=256,pl_location: str="",id=0):
        tokens = self.tokenizer.tokenize(pl_txt)
        remain_tokens_len = max_length - 4 - len(tokens) if len(tokens) < max_length - 4 else 0
        if len(tokens) >= max_length - 4:
            tokens = tokens[:max_length - 4]
            tokens = [self.tokenizer.cls_token, '<encoder-only>', self.tokenizer.sep_token] + tokens + [
                self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_pos = [i+2 for i in range(len(tokens))]
            return {'input_ids': token_ids, 'pos': token_pos, 'attention_mask': np.ones((max_length, max_length),dtype=int)}

        first_map = {}
        if pl_location is not None and pl_location != "":
            first_map = {value.name: value for key, value in self.identifier_tree.items(pl_location)}
        second_map = self.name_map

        # find entity
        tag_code_tokens, entity_list = self.__find_entitys(first_map, second_map, pl_txt)
        tag_sentence = ''.join(tag_code_tokens)
        tag_tokens = self.tokenizer.tokenize(tag_sentence)
        # tag_tokens = ['<s>'] + tag_tokens + ['</s>']
        tokens = []

        # get entity location
        abs_pos = 0
        entity_index = 0
        entity_loc_list = []
        last_tag_location = None
        for index, token in enumerate(tag_tokens):
            if token == self.entity_flag:
                if last_tag_location is None:
                    last_tag_location = abs_pos
                else:
                    entity_loc_list.append(
                        {'entity': entity_list[entity_index], 'location': (last_tag_location+3, abs_pos - 1 + 3)})
                    last_tag_location = None
                    entity_index += 1
            else:
                tokens.append(token)
                abs_pos += 1

        # find relation
        remain_tokens_len = max_length - len(tokens) - 4
        if len(tokens) >= max_length - 4:
            tokens = tokens[0:max_length - 4]
        tokens = [self.tokenizer.cls_token, '<encoder-only>', self.tokenizer.sep_token] + tokens + [
            self.tokenizer.sep_token]
        rels_map = self.find_relation(entity_loc_list, remain_tokens_len,id)

        # split tokens by entity
        rels_locs = list(rels_map.keys())
        rels_locs.sort(key=lambda x: x[0])
        last_split_indice = 0
        split_tokens = []
        for begin, end in rels_locs:
            if begin > last_split_indice:
                split_tokens.append((tokens[last_split_indice:begin], []))
            split_tokens.append((tokens[begin:end + 1], rels_map[(begin, end)]))
            last_split_indice = end + 1
        if last_split_indice < len(tokens):
            split_tokens.append((tokens[last_split_indice:len(tokens)], []))

        if len(split_tokens)>1:
            self.statistics["data_num"]+=1
        
        res =  caculate_base_sent_tree(split_tokens, max_length, self.tokenizer)
        res['tag_tokens'] = tokens
        res['tag_setence']=tag_sentence
        res['input'] = pl_txt
        return res
    def export_set(self,dir:str):
        path = os.path.join(dir,"rel_set")
        rel_list = list(self.rel_count_map.keys())
        with open(path,'w') as f:
            json.dump(rel_list,f)

if __name__ == "__main__":
    data_path = "/Users/zhaowei/code/KEPLMBUG/Input Data/data/WildFlyCore/"
    from transformers import AutoTokenizer

    cbert_model = '/Users/zhaowei/code/KEPLMBUG/Source Code/trace/codebert'
    ctokneizer = AutoTokenizer.from_pretrained(cbert_model, local_files_only=True)
    class MockArgs:
        def __init__(self):
            self.tqdm_interval = 1
            self.code_kg_mode = "other"

    code_knowledge_graph = CodeKnowledgeGraph(data_path, ctokneizer,MockArgs())
    test_pl = '''
    import org.jboss.as.cli.impl.CommandContextConfiguration;
 import org.jboss.as.cli.operation.OperationFormatException;
 OperationFormatException e = new OperationFormatException("test");
 asd.Operate();
    '''
    code_knowledge_graph.add_relation_encode(test_pl,
                                               256,"org.jboss.as.test.integration.management.cli.SecurityCommandsTestCase")
