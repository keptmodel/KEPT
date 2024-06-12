import re
from build_kg_text.data.data_util import *
import markdown
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from build_kg_text.data.data_util import save_string_to_file
# 确保已下载了句子分割器的数据

def to_sentence(content):


# 将Markdown文本转换为HTML
    html = markdown.markdown(content)

# 从HTML中提取纯文本
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

# 使用正则表达式去除所有链接和特殊字符
    text = re.sub(r'http\S+', '', text)  # 去除网址
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # 去除Markdown链接
    text = re.sub(r"[^\w\s\.\,\?\!\;\:$\/\'\"-=<>]", '', text)  # 去除特殊字符，保留句子标点

# 拆分文本为句子
    sentences = sent_tokenize(text)

    return sentences

if __name__=='__main__':
    source_path = "gpt/wildfly"
    output_path = "minie/wildfly"
    task_list=[]


    dir_list = get_all_absdirpath_in_folder(source_path)
    for dir in dir_list:
        file_list = read_all_files_in_folder(dir)
        for filename,content in file_list:
            if filename.endswith('.json'):
                continue
            else:
                data = to_sentence(content)
                save_dir = output_path+"/"+dir.split('/')[-1]
                save_string_to_file('\n'.join(data),save_dir,filename.split('.')[0]+'.txt')
