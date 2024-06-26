import heapq
import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from torch import Tensor

from common.utils import format_batch_input, map_file, map_iss, time_file, time_iss

F_ID = 'id'
F_TOKEN = 'tokens'
F_POS = 'pos'
F_ATTEN_MASK = "attention_mask"
F_INPUT_ID = "input_ids"
F_EMBD = "embd"
F_TK_TYPE = "token_type_ids"

# epoch cache data name
CLASSIFY_RANDON_CACHE = 'classify_random_epoch_{}.cache'
CLASSIFY_NEG_SAMP_CACHE = 'classify_neg_epoch_{}.cache'
RETRIVE_RANDOM_CACHE = 'retrive_random_epoch_{}.cache'
RETRIVE_NEG_SAMP_CACHE = 'retrive_neg_epoch_{}.cache'


def exclude_and_sample(sample_pool, exclude, num):
    for id in exclude:
        sample_pool.remove(id)
    if len(list(sample_pool)) == 0:
        return []
    selected = np.random.choice(list(sample_pool), num)
    return selected


def sample_until_found(sample_pool, exclude, num, retry=3):
    cur_retry = retry
    res = []
    sample_pool = list(sample_pool)
    while num > 0 and retry > 0:
        cho = random.choice(sample_pool)
        if cho in exclude:
            cur_retry -= 1
            continue
        cur_retry = retry
        num -= 1
        res.append(cho)
    return res


def clean_space(text):
    return " ".join(text.split())


class Examples:
    """
    Manage the examples read from raw dataset

    examples:
    valid_examples = CodeSearchNetReader(data_dir).get_examples(type="valid", num_limit=valid_num, summary_only=True)
    valid_examples = Examples(valid_examples)
    valid_examples.update_features(model)
    valid_examples.update_embd(model)

    """

    def __init__(self, raw_examples: List):
        self.__size = 0
        self.NL_index, self.PL_index, self.rel_index, self.rel_file_index = self.__index_exmaple(raw_examples)

    # 按照文件集
    def __is_positive_case(self, nl_id, pl_id):
        nl_id = map_iss[nl_id]
        pl_id = map_file[pl_id]
        if nl_id not in self.rel_file_index:
            return False
        rel_pls = set(self.rel_file_index[nl_id])
        return pl_id in rel_pls

    def __len__(self):
        return self.__size

    def __index_exmaple(self, raw_examples):
        """
        Raw examples should be a dictionary with key "NL" for natural langauge and PL for programming language.
        Each {NL, PL} pair in same dictionary will be regarded as related ones and used as positive examples.
        :param raw_examples:
        :return:
        """
        rel_index = defaultdict(set)
        rel_file_index = defaultdict(set)
        NL_index = dict()  # find instance by id
        PL_index = dict()

        # hanlde duplicated NL and PL with reversed index
        reverse_NL_index = dict()
        reverse_PL_index = dict()

        nl_id_max = 0 if list(map_iss.keys()) == [] else max(list(map_iss.keys())) + 1
        pl_id_max = 0 if list(map_file.keys()) == [] else max(list(map_file.keys())) + 1
        for r_exp in raw_examples:
            pl_tks = r_exp["PL"]
            pl_fid = r_exp['PID']
            if (pl_fid, pl_tks) in reverse_PL_index:
                pl_id = reverse_PL_index[(pl_fid, pl_tks)]
            else:
                reverse_PL_index[(pl_fid, pl_tks)] = pl_id_max
                pl_id = pl_id_max
                pl_id_max += 1
            PL_index[pl_id] = {F_TOKEN: pl_tks, F_ID: pl_id}
            map_file[pl_id] = r_exp['PID']
            time_file[pl_id] = r_exp['PTIME']

            if r_exp["NID"] == -1 or r_exp["NL"] == "NoData" or r_exp["NL"] == "":
                continue

            nl_tks = clean_space(r_exp["NL"])
            if nl_tks in reverse_NL_index:
                nl_id = reverse_NL_index[nl_tks]
            else:
                reverse_NL_index[nl_tks] = nl_id_max
                nl_id = nl_id_max
                nl_id_max += 1
            NL_index[nl_id] = {F_TOKEN: nl_tks, F_ID: nl_id}
            map_iss[nl_id] = r_exp['NID']
            time_iss[nl_id] = r_exp['NTIME']

            if pl_id not in rel_index[nl_id]:
                self.__size += 1
            rel_index[nl_id].add(pl_id)
            rel_file_index[map_iss[nl_id]].add(map_file[pl_id])
        return NL_index, PL_index, rel_index, rel_file_index

    def _gen_feature(self, example, tokenizer):
        feature = tokenizer.encode_plus(example[F_TOKEN], max_length=128, truncation=True,
                                        padding='max_length', return_attention_mask=True,
                                        return_token_type_ids=False)
        res = {
            F_ID: example[F_ID],
            F_INPUT_ID: feature[F_INPUT_ID],
            F_ATTEN_MASK: feature[F_ATTEN_MASK]}
        return res

    def _gen_seq_pair_feature(self, nl_id, pl_id, tokenizer):
        nl_tks = self.NL_index[nl_id][F_TOKEN]
        pl_tks = self.PL_index[pl_id][F_TOKEN]
        feature = tokenizer.encode_plus(
            text=nl_tks,
            text_pair=pl_tks,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            max_length=128,
            add_special_tokens=True
        )
        res = {
            F_INPUT_ID: feature[F_INPUT_ID],
            F_ATTEN_MASK: feature[F_ATTEN_MASK],
            F_TK_TYPE: feature[F_TK_TYPE]
        }
        return res

    def __update_feature_for_index(self, index, tokenizer, n_thread):
        for v in tqdm(index.values(), desc="update feature"):
            f = self._gen_feature(v, tokenizer)
            id = f[F_ID]
            index[id][F_INPUT_ID] = f[F_INPUT_ID]
            index[id][F_ATTEN_MASK] = f[F_ATTEN_MASK]

    def __update_feature_for_code(self, index, tok, n_thread=1):
        for v in tqdm(index.values(), desc="update code feature", maxinterval=3600,
                      mininterval=20):
            f = tok.encoder_plus(v[F_TOKEN])
            id = v[F_ID]
            index[id][F_INPUT_ID] = f[F_INPUT_ID]
            index[id][F_ATTEN_MASK] = f[F_ATTEN_MASK]
            index[id][F_POS] = f[F_POS]

    def __update_feature_for_text(self, index, tok, n_thread=1):
        for v in tqdm(index.values(), desc="update text feature", maxinterval=3600,
                      mininterval=10):
            f = tok.encode_plus(v[F_TOKEN], max_length=256, truncation=True,
                                        padding='max_length', return_attention_mask=True,
                                        return_token_type_ids=False)
            id = v[F_ID]
            index[id][F_INPUT_ID] = f[F_INPUT_ID]
            index[id][F_POS] = None
            index[id][F_ATTEN_MASK] = f[F_ATTEN_MASK]
    def update_features(self, model, n_thread=1):
        """
        Create or overwritten token_ids and attention_mask
        :param model:
        :return:
        """
        #self.__update_feature_for_index(self.PL_index,model.ctokneizer,self.code_kg)
        #self.__update_feature_for_index(self.NL_index,model.ntokenizer,self.text_kg)
        self.__update_feature_for_code(self.PL_index,model.get_pl_tokenizer())
        self.__update_feature_for_text(self.NL_index,model.get_nl_tokenizer())

    def __update_embd_for_index(self, index, func,model ):
        for id in tqdm(index, desc="update embedding", maxinterval=3600, mininterval=30):
            feature = index[id]
            input_tensor = torch.tensor(feature[F_INPUT_ID]).view(1, -1).to(model.device)
            mask_tensor = torch.tensor(feature[F_ATTEN_MASK]).unsqueeze(0).to(model.device)
            pos_tensor = None
            if feature[F_POS] is not None:
                pos_tensor = torch.tensor(feature[F_POS]).view(1, -1).to(model.device)
            embd = func(input_ids=input_tensor, pos=pos_tensor, attention_mask=mask_tensor)
            embd = embd.narrow(1, 0, 1)
            embd_cpu = embd.to('cpu')
            index[id][F_EMBD] = embd_cpu

    def update_embd(self, model):
        """
        Create or overwritten the embedding
        :param model:
        :return:
        """
        with torch.no_grad():
            model.eval()
            self.__update_embd_for_index(self.NL_index, model.create_nl_embd,model)
            self.__update_embd_for_index(self.PL_index, model.create_pl_embd,model)

    def get_retrivial_task_dataloader(self, batch_size):
        """create retrivial task"""
        res = []
        for nl_id in self.NL_index:
            for pl_id in self.PL_index:
                label = 1 if self.__is_positive_case(nl_id, pl_id) else 0
                res.append((nl_id, pl_id, label))
        dataset = DataLoader(res, batch_size=batch_size)
        return dataset

    def get_chunked_retrivial_task_examples(self, chunk_query_num=-1, chunk_size=1000):
        """
        Cut the positive examples into chuncks. For EACH chunk generate queries at a size of query_num * chunk_size
        :param query_num: if query_num is -1 then create queries at a size of chunk_size * chunk_size
        :param chunk_size:
        :return:
        """
        rels = []
        for nid in self.rel_index:
            for pid in self.rel_index[nid]:
                rels.append((nid, pid))
        rel_dl = DataLoader(rels, batch_size=chunk_size)
        examples = []
        for batch in rel_dl:
            batch_query_idx = 0
            nids, pids = batch[0].tolist(), batch[1].tolist()
            for nid in nids:
                batch_query_idx += 1
                if chunk_query_num != -1 and batch_query_idx > chunk_query_num:
                    break
                for pid in pids:
                    label = 1 if self.__is_positive_case(nid, pid) else 0
                    examples.append((nid, pid, label))
        return examples

    def get_chunked_retrivial_task_examples_all(self, chunk_query_num=-1, chunk_size=1000):
        nl_list = list(self.NL_index.keys())
        pl_list = list(self.PL_index.keys())
        examples = []
        for nid in nl_list:
            for pid in pl_list:
                label = 1 if self.__is_positive_case(nid, pid) else 0
                examples.append((nid, pid, label))
        return examples

    def get_chunked_retrivial_task_examples_batch(self, chunk_query_num=-1, chunk_size=1000):
        examples = []
        for nid in self.rel_index:
            for pid in self.rel_index[nid]:
                examples.append((nid, pid, 1))

        nl_list = list(self.NL_index.keys())
        pl_list = list(self.PL_index.keys())
        total = 0
        sample_num = min(chunk_size, len(pl_list))
        for nid in nl_list:
            if total % chunk_size == 0:
                pl_list_chunk = random.sample(pl_list, sample_num)
            total += 1
            for pid in pl_list_chunk:
                if self.__is_positive_case(nid, pid) == 0:
                    examples.append((nid, pid, 0))
        return examples

    def id_pair_to_embd_pair(self, nl_id_tensor: Tensor, pl_id_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert id pairs into embdding pairs"""
        nl_tensor = self._id_to_embd(nl_id_tensor, self.NL_index)
        pl_tensor = self._id_to_embd(pl_id_tensor, self.PL_index)
        return nl_tensor, pl_tensor

    def id_triplet_to_embd_triplet(self, nl_id_tensor, pos_pl_id_tensor, neg_pl_id_tensor):
        nl_tensor = self._id_to_embd(nl_id_tensor, self.NL_index)
        pos_tensor = self._id_to_embd(pos_pl_id_tensor, self.PL_index)
        neg_tensor = self._id_to_embd(neg_pl_id_tensor, self.PL_index)
        return nl_tensor, pos_tensor, neg_tensor

    def id_pair_to_feature_pair(self, nl_id_tensor: Tensor, pl_id_tensor: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert id pairs into embdding pairs"""
        nl_input_tensor, nl_pos_tensor, nl_att_tensor = self._id_to_feature(nl_id_tensor, self.NL_index)
        pl_input_tensor, pl_pos_tensor, pl_att_tensor = self._id_to_feature(pl_id_tensor, self.PL_index)
        return nl_input_tensor, nl_pos_tensor, nl_att_tensor, pl_input_tensor, pl_pos_tensor, pl_att_tensor

    def id_triplet_to_feature_triplet(self, nl_id_tensor, pos_pl_id_tensor, neg_pl_id_tensor):
        nl_input_tensor, nl_att_tensor = self._id_to_feature(nl_id_tensor, self.NL_index)
        pos_pl_input_tensor, pos_pl_att_tensor = self._id_to_feature(pos_pl_id_tensor, self.PL_index)
        neg_pl_input_tensor, neg_pl_att_tensor = self._id_to_feature(neg_pl_id_tensor, self.PL_index)
        return nl_input_tensor, nl_att_tensor, pos_pl_input_tensor, \
            pos_pl_att_tensor, neg_pl_input_tensor, neg_pl_att_tensor

    def _id_to_feature(self, id_tensor: Tensor, index):
        input_ids, input_pos, att_masks = [], [], []
        for id in id_tensor.tolist():
            input_ids.append(torch.tensor(index[id][F_INPUT_ID]))
            if index[id][F_POS] is not None:
                input_pos.append(torch.tensor(index[id][F_POS]))
            if F_ATTEN_MASK in index[id]:
                att_masks.append(torch.tensor(index[id][F_ATTEN_MASK]))
        input_ids_tensor = torch.stack(input_ids)
        input_pos_tensor = None
        if len(input_pos) > 0:
            input_pos_tensor = torch.stack(input_pos)

        att_tensor = None
        if len(att_masks) > 0:
            att_tensor = torch.stack(att_masks)
        return input_ids_tensor, input_pos_tensor, att_tensor


    def _id_to_embd(self, id_tensor: Tensor, index):
        embds = []
        for id in id_tensor.tolist():
            embds.append(index[id][F_EMBD])
        embd_tensor = torch.stack(embds)
        return embd_tensor

    def online_neg_sampling_dataloader(self, batch_size):
        pos = []
        for nl_id in self.rel_index:
            pos_pl_ids = self.rel_index[nl_id]
            for p_id in pos_pl_ids:
                pos.append((nl_id, p_id, 1))
        sampler = RandomSampler(pos)
        dataset = DataLoader(pos, batch_size=batch_size, sampler=sampler)
        return dataset

    def random_neg_sampling_dataloader(self, batch_size):
        pos, neg = [], []
        for nl_id in tqdm(self.rel_index, desc="random_neg_sampling_dataset"):
            pos_pl_ids = self.rel_index[nl_id]
            for p_id in pos_pl_ids:
                pos.append((nl_id, p_id, 1))
            sample_num = len(pos_pl_ids)
            sel_neg_ids = exclude_and_sample(set(self.PL_index.keys()), pos_pl_ids, sample_num)
            for n_id in sel_neg_ids:
                neg.append((nl_id, n_id, 0))
        sampler = RandomSampler(pos + neg)
        dataset = DataLoader(pos + neg, batch_size=batch_size, sampler=sampler)
        return dataset

    def make_online_neg_sampling_batch(self, batch: Tuple, model, hard_ratio):
        """
        Convert positive examples into
        :param batch: tuples  containing the positive examples
        :return:
        """
        nl_ids = batch[0].tolist()
        pl_ids = batch[1].tolist()
        pos, neg = [], []
        cand_neg = []
        for nl_id in nl_ids:
            for pl_id in pl_ids:
                label = 1 if self.__is_positive_case(nl_id, pl_id) else 0
                if label == 0:
                    cand_neg.append((nl_id, pl_id, 0))
                else:
                    pos.append((nl_id, pl_id, 1))

        neg_loader = DataLoader(cand_neg, batch_size=len(batch))
        neg_slots = len(pos)
        hard_neg_slots = max(0, int(hard_ratio * neg_slots))
        rand_neg_slots = max(0, neg_slots - hard_neg_slots)
        if hard_neg_slots>0:
            for neg_batch in neg_loader:
                with torch.no_grad():
                    model.eval()
                    inputs = format_batch_input(neg_batch, self, model)
                    outputs = model(**inputs)
                    logits = outputs['logits']
                    pred = torch.softmax(logits, 1).data.tolist()
                    for nl, pl, score in zip(neg_batch[0].tolist(), neg_batch[1].tolist(), pred):
                        neg.append((nl, pl, score[1]))

        hard_negs = [(x[0], x[1], 0) for x in
                     heapq.nlargest(hard_neg_slots, neg, key=lambda x: x[2])]  # repalce score with label
        rand_negs = []
        if len(cand_neg) != 0:
            rand_negs = random.choices(cand_neg, k=rand_neg_slots)
        res = pos + hard_negs + rand_negs
        random.shuffle(res)
        r_nl, r_pl, r_label = [], [], []
        for r in res:
            r_nl.append(r[0])
            r_pl.append(r[1])
            r_label.append(r[2])
        return (torch.Tensor(r_nl), torch.Tensor(r_pl), torch.Tensor(r_label).long())

    def make_online_triplet_sampling_batch(self, batch: Tuple, model):
        nl_ids = batch[0].tolist()
        pl_ids = batch[1].tolist()
        neg = defaultdict(list)
        cand_neg = []
        res = []
        for nl_id in nl_ids:
            for pl_id in pl_ids:
                label = 1 if self.__is_positive_case(nl_id, pl_id) else 0
                if label == 0:
                    cand_neg.append((nl_id, pl_id, 0))

        neg_loader = DataLoader(cand_neg, batch_size=len(batch))
        for neg_batch in neg_loader:
            with torch.no_grad():
                model.eval()
                nl_embd, pl_embd = format_batch_input(neg_batch, self)
                sim_scores = model.get_sim_score(text_hidden=nl_embd, code_hidden=pl_embd)
                for nl, pl, score in zip(neg_batch[0].tolist(), neg_batch[1].tolist(), sim_scores.tolist()):
                    neg[nl].append((pl, score))

        for nl_id, pl_id in zip(nl_ids, pl_ids):
            if len(neg[nl_id]):
                hard_neg_exmp = heapq.nlargest(1, neg[nl_id], key=lambda x: x[1])[0]
                res.append((nl_id, pl_id, hard_neg_exmp[0]))

        r_nl, r_pos, r_neg = [], [], []
        for r in res:
            r_nl.append(r[0])
            r_pos.append(r[1])
            r_neg.append(r[2])
        return (torch.Tensor(r_nl), torch.Tensor(r_pos), torch.Tensor(r_neg).long())
