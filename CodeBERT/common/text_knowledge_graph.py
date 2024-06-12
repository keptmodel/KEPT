import ahocorasick
import pandas as pd
import spacy
from spacy.symbols import ORTH
from tqdm import tqdm
import numpy as np
from common.code_knowledge_graph import caculate_base_sent_tree


def is_char_split(char):
    return not (char.isalnum() or char == '_' or char == '.')


class TextKnowledgeGraph:
    def __init__(self, data_path: str, tokenize,args):
        self.name_map = {}
        self.args = args
        self.tokenizer = tokenize
        self.name_set = set()
        self.rel_map = {}
        self.A_finder = ahocorasick.Automaton()
        self.nlp = spacy.load('en_core_web_sm')
        self.__load_data(data_path)
        self.statistics = {"rel_num":0,"data_num":0}
        self.entity_flag = '<entity_fuahnr>'
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.entity_flag]})
        
    def __load_data(self, path):
        if path=="":
            return
        data_path = path + 'text_relation.csv'
        rel_pd = pd.read_csv(data_path)
        for index, row in tqdm(rel_pd.iterrows(), total=rel_pd.shape[0], desc='Load Text Triplets',maxinterval = 3600,mininterval=self.args.tqdm_interval):
            head = row['head']
            tail = row['tail']
            rel = row['relation']
            if pd.isna(head) or pd.isna(tail) or pd.isna(rel):
                continue
            if head not in self.name_set:
                self.A_finder.add_word(head, head)
                self.name_set.add(head)
                self.nlp.tokenizer.add_special_case(head, [{ORTH: head}])
            if tail not in self.name_set:
                self.A_finder.add_word(tail, tail)
                self.name_set.add(tail)
                self.nlp.tokenizer.add_special_case(tail, [{ORTH: tail}])

            if head not in self.rel_map:
                self.rel_map[head] = {tail: [rel]}
            else:
                if tail not in self.rel_map[head]:
                    self.rel_map[head][tail] = [rel]
                else:
                    self.rel_map[head][tail].append(rel)
        self.A_finder.make_automaton()

    def add_relation_encode(self, text: str, max_length=256):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) >= max_length - 2:
            tokens = tokens[:max_length - 2]
            tokens = [self.tokenizer.cls_token]+ tokens + [self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_pos = [i+2 for i in range(len(tokens))]
            return {'input_ids': token_ids, 'pos': token_pos, 'attention_mask': np.ones((max_length, max_length),dtype=int)}
        entity_list = []
        entity_set = set()
        tag_text = ''
        doc = self.nlp(text)

        for token in doc:
            if token.text in self.name_set:
                entity_set.add(token.text)
            if token.text in self.rel_map:
                entity_list.append(token.text)
                tag_text += self.entity_flag + token.text + self.entity_flag
            else:
                tag_text += token.text
            tag_text += token.whitespace_ 
        tag_tokens = self.tokenizer.tokenize(tag_text)

        tokens = []
        entity_loc_list = []
        entity_count = 0
        last_index = None
        for token in tag_tokens:
            if token == self.entity_flag:
                if last_index is None:
                    last_index = len(tokens)
                else:
                    entity_loc_list.append(((last_index + 1, len(tokens) - 1 + 1), entity_list[entity_count]))
                    entity_count += 1
                    last_index = None
            else:
                tokens.append(token)

        remain_tokens_len = max_length - len(tokens) - 2
        if len(tokens) > max_length - 2:
            tokens = tokens[0:max_length - 2]
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        relation_map = {}
        for entity_loc, head in entity_loc_list:
            tails = self.rel_map[head]
            for tail in tails:
                if tail in entity_set:
                    rels = self.rel_map[head][tail]
                    for rel in rels:
                        rel_tail = self.tokenizer.tokenize(rel + ' ' + tail)
                        if remain_tokens_len >= len(rel_tail):
                            self.statistics["rel_num"]+=1
                            if entity_loc in relation_map:
                                relation_map[entity_loc].append(rel_tail)
                            else:
                                relation_map[entity_loc] = [rel_tail]
                            remain_tokens_len -= len(rel_tail)

        last_split_indice = 0
        split_tokens = []
        loc_list = list(relation_map.keys())
        loc_list.sort(key=lambda x: x[0])
        for begin, end in loc_list:
            if begin > last_split_indice:
                split_tokens.append((tokens[last_split_indice:begin], []))
            split_tokens.append((tokens[begin:end + 1], relation_map[(begin, end)]))
            last_split_indice = end + 1
        if last_split_indice < len(tokens):
            split_tokens.append((tokens[last_split_indice:len(tokens)], []))

        if len(split_tokens)>1:
            self.statistics["data_num"]+=1
        return caculate_base_sent_tree(split_tokens, max_length, self.tokenizer)


if __name__ == "__main__":
    resource_path = '/Users/zhaowei/code/KEPLMBUG/Input Data/data/WildFlyCore/'
    from transformers import AutoTokenizer

    cbert_model = '/Users/zhaowei/Downloads/kebug_unix/trace/unixCoder'
    ctokneizer = AutoTokenizer.from_pretrained(cbert_model, local_files_only=True)
    import parser


    class MockArgs:
        def __init__(self):
            self.tqdm_interval = 10
    text_kg = TextKnowledgeGraph(resource_path, ctokneizer, MockArgs())
    test_text = '''
    system properties should not be here.
    youshould jboss.bind.address
    and HA functionality
    '''
    text_kg.add_relation_encode(test_text)
