import pandas as pd
from build_kg_text.data.data_util import *
import json
import nltk
from nltk.tokenize import word_tokenize


def is_valid(sent):
    if '.' in sent:
        return True
    tokens = word_tokenize(sent)
    tagged = nltk.pos_tag(tokens)
    has_noun = any(tag for word, tag in tagged if tag.startswith('N'))
    has_unknown = any(tag for word, tag in tagged if tag == 'NN')
    has_cd = any(tag for word, tag in tagged if tag == 'CD')
    has_adj = any(tag for word, tag in tagged if tag.startswith('JJ'))
    if not has_noun and not has_unknown and not has_cd and not has_adj:
        for word, tag in tagged:
            print(word + ":" + tag)
        print(sent)
        print("====")
    return has_noun or has_unknown or has_cd or has_adj




def get_triplets(data):
    if len(data) == 3 and type(data[0]) == str:
        return [data]
    if len(data) < 3:
        return []
    res = []
    for item in data:
        if item != None:
            res.extend(get_triplets(item))
    return res

data_path = "mineie/log4j2"
task_list = get_all_absdirpath_in_folder(data_path)
rels_data = []
for item in task_list:
    file_list = read_all_files_in_folder(item)
    for filename, content in file_list:
        if filename.endswith('.json'):
            data = json.loads(content)
            data = get_triplets(data)
            rels_data.extend(data)

checked_rels_data = []

for rel in rels_data:
    try:
        if not is_valid(rel[0]) or not is_valid(rel[2]):
            continue
        checked_rels_data.append(rel)
    except Exception as e:
        print(e)
        print(rel)
        continue

df = pd.DataFrame(checked_rels_data, columns=['head', 'relation', 'tail'])

output = data_path+"/text_relation.csv"
output = get_resource_path(output)
df.to_csv(output)
