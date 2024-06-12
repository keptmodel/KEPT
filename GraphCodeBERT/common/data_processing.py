import gzip
import json
import os
from collections import defaultdict
from pathlib import Path


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


class CodeSearchNetReader:
    def __init__(self, data_dir, lang="python"):
        self.data_dir = data_dir
        self.is_training = True
        self.lang = lang

    def get_summary_from_docstring(self, docstring):
        summary = []
        for line in docstring.split("\n"):
            if self.lang == 'python':
                clean_line = line.strip("\n\t\r \"")
                if len(clean_line) == 0:
                    break
                if clean_line.startswith(":") or clean_line.startswith("TODO") \
                        or clean_line.startswith("Parameter") or clean_line.startswith("http"):
                    break
                summary.append(clean_line)
            else:
                summary.append(line)
        return " ".join(summary)

    def get_examples(self, type, num_limit=None, repos=[], summary_only=True):
        """
        :param type: train, valid, test
        :param num_limit: max number of examples
        :return:
        """
        examples = []
        doc_dup_check = defaultdict(list)
        json_dir = os.path.join(self.data_dir, "final/jsonl")
        src_files = Path(os.path.join(json_dir, type)).glob('*.gz')
        for zfile in src_files:
            print("processing {}".format(zfile))
            if num_limit is not None:
                if num_limit <= 0:
                    break
            with gzip.open(zfile, 'r') as fin:
                for line in fin.readlines():
                    if num_limit is not None:
                        if num_limit <= 0:
                            break
                    jobj = json.loads(str(line, encoding='utf-8'))
                    repo = jobj['repo']
                    if len(repos) > 0 and repo not in repos:
                        continue
                    # code = jobj['code']
                    code = ' '.join([format_str(token) for token in jobj['code_tokens']])
                    # doc_str = jobj['docstring']
                    doc_str = ' '.join(jobj['docstring_tokens'])
                    code = code.replace(doc_str, "")
                    if summary_only:
                        doc_str = self.get_summary_from_docstring(doc_str)
                    if len(doc_str.split()) < 10:  # abandon cases where doc_str is shorter than 10 tokens
                        continue
                    if num_limit:
                        num_limit -= 1
                    example = {
                        "NL": doc_str,
                        "PL": code
                    }
                    doc_dup_check[doc_str].append(example)
                    if num_limit and len(doc_dup_check[doc_str]) > 1:
                        num_limit += 1 + (len(doc_dup_check[doc_str]) == 2)

        for doc in doc_dup_check:
            if len(doc_dup_check[doc]) > 1:
                continue
            examples.extend(doc_dup_check[doc])
        return examples  # {nl:[pl]}
import json
import os

import pandas as pd
import argparse

def get_hunks_from_diff(diff:str):
    lines = diff.split('\n')
    hunks = []
    current_hunk = []


    for line in lines:

        if line.startswith('@@'):

            if current_hunk:
                hunks.append('\n'.join(current_hunk))
                current_hunk = []

        current_hunk.append(line)


    if current_hunk:
        hunks.append('\n'.join(current_hunk))
    return hunks

def read_all_files_in_folder(folder_path):

    files = os.listdir(folder_path)

    result = {}

    for file_name in files:

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as file:
            data = file.read()

        data = json.loads(data)
        result[file_name.split('.')[0]] = data
    return result



def mkdir(path):

    path = path.strip()
    isExists = os.path.exists(path)


    if not isExists:

        os.makedirs(path)
        return True
    else:

        return False


def process_location(location: str):
    ori_location =location
    if 'org/jboss' in location:
        location="org/jboss/"+location.split('org/jboss/')[1]
    elif 'org/apache' in location:
        location="org/apache/"+location.split('org/apache/')[1]
    elif 'src/main/java/' in location:
        location = location.split('src/main/java/')[1]
    elif 'src/test/java/' in location:
        location = location.split('src/test/java/')[1]
    elif 'src/main/' in location:
        location = location.split('src/main/')[1]
    elif 'src/test/' in location:
        location = location.split('src/test/')[1]
    elif 'classes/' in location:
        location = location.split('classes/')[1]
    elif 'src/' in location:
        location = location.split('src/')[1]
    else:
        location = ''
    location = location.replace('/', '.')
    location = location.rsplit('.', 1)[0]
    return location

def add_diff_content(data_map,location,file_diff_id_count,commit_time,diff_content,issue_key,commit_id_count,file_id_count):
    links=data_map["links"]
    file_diffs=data_map["file_diffs"]
    file_locations=data_map["file_locations"]
    file_commit_times=data_map["file_commit_times"]
    reverser_diff_map=data_map["reverser_diff_map"]
    commit_id_map = data_map["commit_id_map"]
    file_id_map = data_map["file_id_map"]
    if diff_content in reverser_diff_map:
        diff_id = reverser_diff_map[diff_content]
        if issue_key in links.keys():
            links[issue_key].append(diff_id)
        else:
            links[issue_key] = [diff_id]
    else:
        reverser_diff_map[diff_content] = file_diff_id_count
        file_diffs[file_diff_id_count] = diff_content

        file_locations[file_diff_id_count] = location
        commit_id_map[file_diff_id_count] = commit_id_count
        file_id_map[file_diff_id_count] = file_id_count

        file_commit_times[file_diff_id_count] = commit_time
        if issue_key in links.keys():
            links[issue_key].append(file_diff_id_count)
        else:
            links[issue_key] = [file_diff_id_count]
        file_diff_id_count += 1
    return file_diff_id_count

def process_data(issue_path, commit_path, data_type):
    """
    Process the data and return the processed data
    :param data_path: the path of the data
    :return: the processed data
    """
    issue_pd = pd.read_csv(issue_path)
    commits = read_all_files_in_folder(commit_path)
    issue_pd = issue_pd[issue_pd['issue_id'].isin(commits.keys())]
    commits = {k: commits[k] for k in issue_pd['issue_id'].values}

    reverser_diff_map = {}
    links = {}
    file_diffs = {}
    file_locations = {}
    file_diff_id_count = 1
    commit_id_count = 1
    file_id_count = 1
    commit_id_map = {}
    file_id_map ={}
    file_commit_times = {}
    data_map = {
        "links":links,
        "file_diffs":file_diffs,
        "file_locations":file_locations,
        "file_commit_times":file_commit_times,
        "reverser_diff_map":reverser_diff_map,
        "commit_id_map":commit_id_map,
        "file_id_map":file_id_map
    }
    for issue_key, value in commits.items():
        for commit in value:
            if 'commit' not in commit.keys():
                continue
            commit_time = commit['commit']['committer']['date']
            commit_content = ''
            for file in commit['files']:
                location = file['filename']
                location = process_location(location)
                if 'patch' not in file.keys():
                    continue
                if data_type=="data_hunk":
                    contents = get_hunks_from_diff(file['patch'])
                    for ct in contents:
                        file_diff_id_count = add_diff_content(data_map,location,file_diff_id_count,commit_time,ct,issue_key,commit_id_count,file_id_count)
                elif data_type=="data_file":
                    file_diff_id_count = add_diff_content(data_map,location,file_diff_id_count,commit_time,file['patch'],issue_key,commit_id_count,file_id_count)
                else:
                    commit_content+=file['patch']
                file_id_count+=1
            if data_type=="data_commit":
                if len(commit_content)>0:
                    file_diff_id_count = add_diff_content(data_map,"",file_diff_id_count,commit_time,commit_content,issue_key,commit_id_count,file_id_count)
            commit_id_count+=1


    links_list = []
    for issue_id, file_id_list in links.items():
        for file_id in file_id_list:
            links_list.append({'bug_report_id': issue_id, 'file_diff_id': file_id})

    file_diffs_list = []
    for file_diff_id in range(1, file_diff_id_count):
        file_diffs_list.append({'file_diff_id': file_diff_id, 'commit_time': file_commit_times[file_diff_id],
                                'file_diff': file_diffs[file_diff_id], 'file_location': file_locations[file_diff_id],"commit_id":commit_id_map[file_diff_id],"file_id":file_id_map[file_diff_id]})

    bug_reports_list = []
    for index, row in issue_pd.iterrows():
        if pd.isna(row['description']):
            continue
        bug_reports_list.append({'bug_report_id': row['issue_id'], 'bug_report_desc': row['description'],
                                 'bug_report_time': row['created_date']})

    bug_reports_pd = pd.DataFrame(bug_reports_list)
    links_pd = pd.DataFrame(links_list)
    file_diffs_pd = pd.DataFrame(file_diffs_list)
    links_pd = links_pd[links_pd["bug_report_id"].isin(bug_reports_pd["bug_report_id"])]
    file_diffs_pd = file_diffs_pd[file_diffs_pd["file_diff_id"].isin(links_pd["file_diff_id"])]
    return bug_reports_pd,links_pd,file_diffs_pd
    

TRAIN_RATIO = 0.5


def divide_dataset(bug_reports_pd: pd.DataFrame, links_pd: pd.DataFrame, file_diffs_pd: pd.DataFrame):
    # 转换bug_reports_pd中commit_time为时间类型
    bug_reports_pd['bug_report_time'] = pd.to_datetime(bug_reports_pd['bug_report_time'])
    bug_reports_pd = bug_reports_pd.sort_values('bug_report_time').reset_index(drop=True)
    train_bug_reports_pd = bug_reports_pd[:int(len(bug_reports_pd) * TRAIN_RATIO)].reset_index(drop=True)
    test_bug_reports_pd = bug_reports_pd[int(len(bug_reports_pd) * TRAIN_RATIO):].reset_index(drop=True)
    train_links_pd = links_pd[links_pd['bug_report_id'].isin(train_bug_reports_pd['bug_report_id'])].reset_index(drop=True)
    test_links_pd = links_pd[links_pd['bug_report_id'].isin(test_bug_reports_pd['bug_report_id'])].reset_index(drop=True)
    train_file_diffs_pd = file_diffs_pd[file_diffs_pd['file_diff_id'].isin(train_links_pd['file_diff_id'])].reset_index(drop=True)
    #test_file_diffs_pd = file_diffs_pd[file_diffs_pd['file_diff_id'].isin(test_links_pd['file_diff_id'])]
    test_file_diffs_pd = file_diffs_pd.reset_index(drop=True)
    return (train_bug_reports_pd, train_links_pd, train_file_diffs_pd), (
        test_bug_reports_pd, test_links_pd, test_file_diffs_pd)

def process_project(base_dir,project,data_type="data_hunk"):
    issue_path = os.path.join(base_dir,f"raw/issue/{project}.csv")
    commit_path = os.path.join(base_dir,f"raw/commit/{project}/")
    output_path = os.path.join(base_dir,f"{project}/")
    bug_reports_pd, links_pd, file_diffs_pd = process_data(issue_path, commit_path,data_type)
    train_dataset, test_dataset = divide_dataset(bug_reports_pd, links_pd, file_diffs_pd)
    mkdir(output_path + "train")
    mkdir(output_path + "test")
    train_dataset[0].reset_index(drop=True).to_csv(output_path + "train/bug_reports.csv")
    train_dataset[1].reset_index(drop=True).to_csv(output_path + "train/links.csv")
    train_dataset[2].reset_index(drop=True).to_csv(output_path + "train/file_diffs.csv")
    test_dataset[0].reset_index(drop=True).to_csv(output_path + "test/bug_reports.csv")
    test_dataset[1].reset_index(drop=True).to_csv(output_path + "test/links.csv")
    test_dataset[2].reset_index(drop=True).to_csv(output_path + "test/file_diffs.csv")


def preprocess_dataset(base_dir,project):
    process_project(base_dir,project)
