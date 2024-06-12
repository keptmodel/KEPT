import random

import pandas as pd
import os
from data.data_util import *
import json
from collections import defaultdict
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

def process_doc_list(path,data_set:set):
    count = 0
    with open(path,'w') as file:
        for item in data_set:
            file.write(str(count)+'\t'+item+'\n')
            count+=1
def process_issue_time(path,map:dict):
    with open(path,'w') as file:
        for key in map:
            file.write(str(key)+','+str(map[key])+'\n')
def pick_random_negative(issue_key2diff,key,name2content):
    file_name_set = issue_key2diff[key]
    file_diff_set = set()
    for file in file_name_set:
        file_diff_set.add(name2content[file])
    while True:
        
        random_key = random.choice(list(issue_key2diff.keys()))
        if len(issue_key2diff[random_key])<=0 or random_key==key:
            continue
        random_file = random.choice(list(issue_key2diff[random_key]))
        diff1 = name2content[random_file]
        if diff1 not in file_diff_set:
            return random_file
def process_issue_git(path,map:dict):
    data_list = []
    for key in map:
        for value in map[key]:
            data_list.append({"issue":key,"repo":0,"sha":value[:7]})
    pd.DataFrame(data_list).to_csv(path,index=False,sep='\t')
def process_data(project_name):
    issues_df =pd.read_csv(get_abs_path(f"issue",f"{project_name}.csv"))
    commit_content = read_all_files_in_folder(f"commit/{project_name}")
    issues_df = issues_df[issues_df['issue_id'].isin(commit_content.keys())]
    issues_df = issues_df[issues_df["description"].notnull()]
    commit_content = {k: commit_content[k] for k in issues_df['issue_id'].values}

    hunk_set=set()
    file_set=set()
    commit_set=set()

    issue2hash=defaultdict(list)

    issue_key2file_c=defaultdict(set)
    issue_key2hunk_c =defaultdict(set)
    issue_key2commit_c =defaultdict(set)

    name2content={}
    commit_count=0
    tmp_set = set()

    for issue_key, commits in commit_content.items():
        #commits = json.loads(content)
        for commit in commits:
            if 'commit' not in commit.keys():
                continue
            file_count=0
            hunk_count=0
            commit_time = commit['commit']['committer']['date']
            #convert commit_time into int time format
            commit_time = pd.to_datetime(commit_time)
            commit_time = int(commit_time.timestamp())
            log = commit['commit']['message']
            sha5 = commit['sha']
            commit_diff = ""
            for file in commit['files']:
                location = file['filename']
                if 'patch' not in file:
                    continue
                file_diff = file['patch']
                if len(file_diff)<=0:
                    continue
                hunk_diffs = get_hunks_from_diff(file_diff)
                file_diff = "+++ "+location+"\n"
                for hunk in hunk_diffs:
                    hunk_name="c_"+sha5+"_"+str(hunk_count)
                    hunk_count+=1
                    file_diff += hunk
                    hunk = "+++ "+location+"\n"+hunk
                    hunk_dic={
                        "sha":sha5,
                        "log":log,
                        "commit":hunk,
                        "timestamp":commit_time
                              }
                    hunk_str = json.dumps(hunk_dic)
                    hunk_set.add(f"{hunk_name}.json")
                    issue_key2hunk_c[issue_key].add(f"{hunk_name}.json")
                    name2content[f"{hunk_name}.json"]=hunk
                    save_string_to_file(hunk_str,f"{project_name}/hunks",f"{hunk_name}.json")
                file_name = "c_"+sha5+"_"+str(file_count)
                file_count+=1
                commit_diff += file_diff
                file_dic={
                    "sha": sha5,
                    "log": log,
                    "commit": file_diff,
                    "timestamp": commit_time
                }
                file_set.add(f"{file_name}.json")
                issue_key2file_c[issue_key].add(f"{file_name}.json")
                name2content[f"{file_name}.json"]=file_diff
                save_string_to_file(json.dumps(file_dic),f"{project_name}/files",f"{file_name}.json")
            commit_name="c_"+sha5+"_0"
            commit_dic={
                "sha": sha5,
                "log": log,
                "commit": commit_diff,
                "timestamp": commit_time
            }
            if len(commit_diff)<=0:
                #print("1")
                continue
            commit_count+=1
            tmp_set.add(commit_diff)
            issue2hash[issue_key].append(sha5)
            commit_set.add(f"{commit_name}.json")
            issue_key2commit_c[issue_key].add(f"{commit_name}.json")
            name2content[f"{commit_name}.json"]=commit_diff
            save_string_to_file(json.dumps(commit_dic),f"{project_name}/commits",f"{commit_name}.json")
    print(commit_count)
    print(len(tmp_set))
    issues_df['summary'] = issues_df['summary'].apply(lambda x: " ".join(x.split(" ")[2:]))
    issues_df=issues_df[issues_df["issue_id"].isin(issue_key2hunk_c.keys())]
    issue_count = 1
    issue_map = {}
    issue2fixtime = {}
    issue2opentime={}
    issue2git=defaultdict(list)



    for index,row in issues_df.iterrows():
        issue_id = row["issue_id"]
        issue_map[issue_count] = issue_id
        save_string_to_file(row["summary"],f"{project_name}/br/short",f"{issue_count}.txt")
        save_string_to_file(row["description"],f"{project_name}/br/long",f"{issue_count}.txt")

        fix_time = pd.to_datetime(row['fixed_date'])
        open_time = pd.to_datetime(row['created_date'])
        fix_time = int(fix_time.timestamp())
        open_time = int(open_time.timestamp())
        issue2fixtime[issue_count] = fix_time
        issue2opentime[issue_count] = open_time

        issue2git[issue_count].extend(issue2hash[issue_id])

        issue_count+=1


    process_doc_list(get_abs_path(f"{project_name}","doc_list_commits.tsv"),commit_set)
    process_doc_list(get_abs_path(f"{project_name}","doc_list_files.tsv"),file_set)
    process_doc_list(get_abs_path(f"{project_name}","doc_list_hunks.tsv"),hunk_set)

    process_issue_time(get_abs_path(f"{project_name}","fix_ts.txt"),issue2fixtime)
    process_issue_time(get_abs_path(f"{project_name}","open_ts.txt"),issue2opentime)

    process_issue_git(get_abs_path(f"{project_name}","issue2git.tsv"),issue2git)

    # 将created_date转换为日期时间格式
    issues_df['created_date'] = pd.to_datetime(issues_df['created_date'])

    # 根据created_date进行排序
    issues_df = issues_df.sort_values(by='created_date')

    # 选择前一半的数据
    half_length = len(issues_df) // 2
    issues_df = issues_df.iloc[:half_length]

    train_list_hunk=[]
    train_list_commit=[]
    train_list_file=[]
    for index,row in issues_df.iterrows():
        issue_key=row["issue_id"]
        content = row["summary"]+" "+row["description"]
        content=content.replace(","," ")
        for diff in issue_key2commit_c[issue_key]:
            train_list_commit.append({
                "log":content,
                "hunk":diff,
                "label":"1.0"
            })
            train_list_commit.append({
                "log":content,
                "hunk":pick_random_negative(issue_key2commit_c,issue_key,name2content),
                "label":"0.0"
            })
        for diff in issue_key2file_c[issue_key]:
            train_list_file.append({
                "log":content,
                "hunk":diff,
                "label":"1.0"
            })
            train_list_file.append({
                "log":content,
                "hunk":pick_random_negative(issue_key2file_c,issue_key,name2content),
                "label":"0.0"
            })
        for diff in issue_key2hunk_c[issue_key]:
            train_list_hunk.append({
                "log":content,
                "hunk":diff,
                "label":"1.0"
            })
            train_list_hunk.append({
                "log":content,
                "hunk":pick_random_negative(issue_key2hunk_c,issue_key,name2content),
                "label":"0.0"
            })
    pd.DataFrame(train_list_commit).to_csv(get_abs_path(f"{project_name}","training_dataset_RN_commits.csv"),index=False)
    pd.DataFrame(train_list_file).to_csv(get_abs_path(f"{project_name}","training_dataset_RN_files.csv"),index=False)
    pd.DataFrame(train_list_hunk).to_csv(get_abs_path(f"{project_name}","training_dataset_RN_hunks.csv"),index=False)
    return

PROJECT_LIST=[
    "hornetq",
    "seam2",
    "weld",
    "teiid",
    "drools",
    "derby",
    "log4j2"
]
if __name__ == '__main__':
    for project in PROJECT_LIST:
        process_data(project)
    print("done")







