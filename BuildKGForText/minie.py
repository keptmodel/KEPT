#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:08:18 2018

@author: Emmanouil Theofanis Chourdakis <e.t.chourdakis@qmul.ac.uk>

This re-implements src/main/java/tests/minie/Demo.java with python
in order to showcase how minie can be used with python.

"""
import json
# Change CLASSPATH to point to the minie jar archive,

from tqdm import tqdm

from build_kg_text.data.data_util import *
from build_kg_text.to_sentence import to_sentence

os.environ['CLASSPATH'] = "/Users/zhaowei/code/miniepy/target/minie-0.0.1-SNAPSHOT.jar"
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/jdk-1.8.jdk/Contents/Home'
# Uncomment to point to your java home (an example is given for arch linux)
# if you don't have it as an environment variable already.
# os.environ['JAVA_HOME'] = '/usr/lib/jvm/default'

# Import java classes in python with pyjnius' autoclass (might take some time)
from jnius import autoclass

CoreNLPUtils = autoclass('de.uni_mannheim.utils.coreNLP.CoreNLPUtils')
AnnotatedProposition = autoclass('de.uni_mannheim.minie.annotation.AnnotatedProposition')
MinIE = autoclass('de.uni_mannheim.minie.MinIE')
StanfordCoreNLP = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
String = autoclass('java.lang.String')

# Dependency parsing pipeline initialization
parser = CoreNLPUtils.StanfordDepNNParser()


# Input sentence
def extract_relation(sentence_list):
    triplet_list=[]
    for sentence in sentence_list:
        try:
            minie = MinIE(String(sentence), parser, 2)
            for ap in minie.getPropositions().elements():
                if ap is not None:
                    triple = ap.getTriple()
                    if triple.size()==3:
                        triplet_list.append([triple.get(0).getWords(),triple.get(1).getWords(),triple.get(2).getWords()])
        except Exception as e:
            print(e)
    return triplet_list

if __name__=='__main__':
    source_path = "minie/wildfly"
    task_list=[]


    dir_list = get_all_absdirpath_in_folder(source_path)
    for dir in dir_list:
        file_list = read_all_files_in_folder(dir)
        file_set = set([file[0] for file in file_list])
        if 'data.json' in file_set:
            continue
        for filename,content in file_list:
            if (filename.split('.')[0]+'.json') in file_set:
                continue
            else:
                task_list.append((dir,filename,content))

    for dir,filename,content in tqdm(task_list,total=len(task_list),desc='Processing'):
        sent_list = to_sentence(content)
        triplets = extract_relation(sent_list)
        save_string_to_file(json.dumps(triplets),dir,filename.split('.')[0]+'.json')
