import xml.etree.ElementTree as ET
from data.data_util import *
from datetime import datetime
from datetime import datetime
import pytz
import json
import os
import pandas as pd

def write_colum(path,datalist:list):
    with open(path,'w') as file:
        for item in datalist:
            file.write("\t".join(item)+'\n')
def write_text(path,datalist:list):
    with open(path,'w') as file:
        for item in datalist:
            file.write(item)

def formate_commit(commit_data):
    dt_utc = datetime.strptime(commit_data['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
    formatted_str = dt_utc.strftime('%a %b %d %H:%M:%S %Y +0000')

    output = f"commit {commit_data['sha']}\n"
    output +=f"Author: {commit_data['commit']['author']['name']} <{commit_data['commit']['author']['email']}>\n"
    output +=f"Date:   {formatted_str}\n\n"
    output += f"     \n"
    return output

def format_diff(commit_data):
    dt_utc = datetime.strptime(commit_data['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
    formatted_str = dt_utc.strftime('%a %b %d %H:%M:%S %Y +0000')

    output = f"commit {commit_data['sha']}\n"
    output +=f"Author: {commit_data['commit']['author']['name']} <{commit_data['commit']['author']['email']}>\n"
    output +=f"Date:   {formatted_str}\n\n"
    output += f"     \n"
    for file in commit_data['files']:
        if 'previous_filename' not in file:
            file['previous_filename'] = file['filename']
        output += f"diff --git a/{file['previous_filename']} b/{file['filename']}\n"
        output += f"index 00000000..00000000\n"
        output += f"--- a/{file['previous_filename']}\n"
        output += f"+++ b/{file['filename']}\n"
        if 'patch' in file:
            output += f"{file['patch']}\n"

    return output

def process_data(project_name):
    output_path = f"locus/{project_name}"
    root = ET.Element("bugrepository")
    root.set("name", project_name)

    issue_path = get_abs_path(f"issue",f"{project_name}.csv")
    issues_df = pd.read_csv(issue_path)
    issues_df["time_rank"] = pd.to_datetime(issues_df["created_date"])
    issues_df = issues_df.sort_values(by="time_rank")
    issues_df = issues_df.dropna(subset=['description'])

    half_length = len(issues_df) // 2
    issues_df = issues_df.iloc[half_length:]

    file_list = read_all_files_in_folder(f"commit/{project_name}")
    file_list = [(x.split('.')[0],y) for x,y in file_list]
    issue_commit_map = {x:json.loads(y) for x,y in file_list}
    for index, row in issues_df.iterrows():
        id = str(row['issue_id']).split('-')[-1]
        dt = datetime.strptime(row['created_date'], "%Y-%m-%dT%H:%M:%SZ")
        opendate = dt.strftime("%Y-%m-%d %H:%M:%S")
        dt = datetime.strptime(row['fixed_date'], "%Y-%m-%dT%H:%M:%SZ")
        fixdate = dt.strftime("%Y-%m-%d %H:%M:%S")
        bug = ET.SubElement(root, "bug", id=id,opendate=opendate,fixdate=fixdate)
        buginformation = ET.SubElement(bug, "buginformation")
        ET.SubElement(buginformation, "summary").text = ""
        if pd.isna(row['description']):
            ET.SubElement(buginformation, "description").text = ""
        else:
            ET.SubElement(buginformation, "description").text = row['description']

        #fixedfiles=ET.SubElement(bug, "fixedFiles")
        relativeCommit = ET.SubElement(bug,"relativeCommit")
        for commit in issue_commit_map[row['issue_id']]:
            ET.SubElement(relativeCommit, "commit").text = commit['sha'][0:8]
    tree = ET.ElementTree(root)
    bugrepo_path = output_path+"/bugrepo"
    check_and_create_path(bugrepo_path)
    print(get_abs_path(bugrepo_path,f"repository.xml"))
    tree.write(get_abs_path(bugrepo_path,f"repository.xml"),encoding="utf-8",xml_declaration=True)

    commit_data=[]
    full_commit_date = []
    for key,commit_list in issue_commit_map.items():
        for commit in commit_list:
            row = []

            row.append(commit['sha'][0:8])
            row.append(commit["commit"]["author"]['name'])

            dt_utc = datetime.strptime(commit['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
            formatted_str = dt_utc.strftime('%a %b %d %H:%M:%S %Y +0000')
            row.append(formatted_str)

            row.append(" ")
            commit_data.append(row)
            full_commit_date.append(formate_commit(commit))

            sha5 = commit['sha'][0:8]
            dir = f"locus/{project_name}/code/{sha5[0:2]}/{sha5[2:4]}"
            check_and_create_path(f"code/{sha5[0:2]}/{sha5[2:4]}")
            diff_content = format_diff(commit)
            save_string_to_file(diff_content,dir,f"{sha5}.txt")
    write_colum(get_abs_path(output_path,"logOneline.txt"),commit_data)
    write_text(get_abs_path(output_path,"logFullDescription.txt"),full_commit_date)

PROJECT_LIST=[
    "hornetq",
    "seam2",
    "weld",
    "teiid",
    "derby",
    "drools",
    "log4j2"
]
if __name__=="__main__":
    for pt in PROJECT_LIST:
        process_data(pt)
    print("done")



