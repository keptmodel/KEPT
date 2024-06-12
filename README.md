# KEPT-replication-package

This repository contains source code that we used to perform experiment in paper titled "A Knowledge Enhanced Pre-trained Model for Bug Localization".

The structure of this folder is as follows：

- KEPT: Contains the source code to reproduce the result of RAPT model.
- CodeT5: Contains the source code to reproduce the result of CodeT5 baseline.
- Gpt2: Contains the source code to reproduce the result of GPT2 baseline.
- Locus: Contains the source code to reproduce the result of Locus baseline.
- GraphCodeBERT: Contains the source code to reproduce the result of GraphCodeBERT baseline.
- SemanticCodeBERT: Contains the source code  to reproduce the result of SemanticCodeBERT.
- CodeBERT: Contains the source code to reproduce the result of CodeBERT.
- Bert: Contains the source code to reproduce the result of BERT.
- Kept_Pretrain: Contains the source code of pre-training for KEPT model.
- BuildKGForText: Contains tools for building knowledge graphs from project document.
- BuildKGForCode: Contains tools for building knowledge graphs from code.

For some large models and dataset, please download from the link https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing.

## Environment Requirements

* Python >= 3.8 
* pytorch==1.11.0
* 1 GPU with CUDA 11.5
* Java Development Kit == 1.8
* Maven == 3.9.6

GPU used in experiments: Nvidia A100-PCIE-40GB

GPU used in SemanticCodeBERT:Nvidia GTX 3090
## KEPT

### Dataset

The data we used to train and test is attached in link https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing.

Data.zip is the KEPT dataset. You need to download it and then extract it to any location.


Data Structure: 

XX is the name of the project.

raw/issue/XX.csv: Bug report data of the project.
raw/commit/XX: Changeset data of the project.

XX/text_relation.csv: Knowledge graph generated from the project documentation.

XX/code_relation.csv: Relationships contained in the knowledge graph generated from the project code.

XX/code_entity.csv: Entities contained in the knowledge graph generated from the project code.

### Experiment


Before starting the experiment, there are some steps to follow, 

Step1: Install the python dependencies listed in the requirements.txt file within the directory KEPT.

Step2: Download the KEPT.zip file from https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing, then unzip it, and place the rapt.pt file in trace/main/model, and the remaining files in the /trace/unixCoder folder

Step3:Modify the DATA_SOURCE in trace/main/train_eval.sh to point to your data dir.

Step4: Modify the PROJECT_NAME in trace/main/train_eval.sh to point to the project you want to train and evaluate
#### Train & Evaluation

We provide a script that combines the training and evaluation processes. You can run it to reproduce our experiments.

The experimental results are in the trace/main/result folder
```bash
cd trace/main
bash train_eval.sh
```


## Gpt2

### Dataset

The same with KEPT

### Experiment


Before starting the experiment, there are some steps to follow, 

Step1: Install the python dependencies listed in the requirements.txt file within the directory Gpt2.

Step2: Download the GPT2.zip file from https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing, then unzip it, and place all files in the /trace/GPT2 folder.

Step3:Modify the DATA_SOURCE in trace/main/train_eval.sh to point to your data dir.

Step4: Modify the PROJECT_NAME in trace/main/train_eval.sh to point to the project you want to train and evaluate
#### Train & Evaluation

We provide a script that combines the training and evaluation processes. You can run it to reproduce our experiments.

The experimental results are in the trace/main/result folder
```bash
cd trace/main
bash train_eval.sh
```


## CodeT5

### Dataset

Same as KEPT.


### Experiment


Before starting the experiment, there are some steps to follow, 

Step1: Install the python dependencies listed in the requirements.txt file within the directory CodeT5.

Step2: Download the CodeT5.zip file from https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing, then unzip it, and place all files in the /trace/CodeT5 folder.

Step3:Modify the DATA_SOURCE in trace/main/train_eval.sh to point to your data dir.

Step4: Modify the PROJECT_NAME in trace/main/train_eval.sh to point to the project you want to train and evaluate
#### Train & Evaluation

We provide a script that combines the training and evaluation processes. You can run it to reproduce our experiments.

The experimental results are in the trace/main/result folder
```bash
cd trace/main
bash train_eval.sh
```
## GraphCodeBERT

### Dataset

Same as KEPT.


### Experiment


Before starting the experiment, there are some steps to follow, 

Step1: Install the python dependencies listed in the requirements.txt file within the directory GraphCodeBERT.

Step2: Download the GraphCodeBERT.zip file from https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing, then unzip it, and place all files in the /trace/gcbert folder.

Step3:Modify the DATA_SOURCE in trace/main/train_eval.sh to point to your data dir.

Step4: Modify the PROJECT_NAME in trace/main/train_eval.sh to point to the project you want to train and evaluate
#### Train & Evaluation

We provide a script that combines the training and evaluation processes. You can run it to reproduce our experiments.

The experimental results are in the trace/main/result folder
```bash
cd trace/main
bash train_eval.sh
```
## Locus

### Dataset

The data we used to train and test is attached in link https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing.

Data_Locus.zip is the Locus dataset. You need to download it and then extract it to any location. Data_Locus.zip is converted from Data.zip, there is a convert tool in code too.
Then, modify the config_example.txt in trace/main/train_eval.sh to point to your data dir.
You can get more information in the Locus Readme
### Experiment
Before the experiment, you need to compile the Locus executable file. Locus is built and developed through eclipse. You need to build it through the following steps
- Open Eclipse
- Make a 'techniques' folder into workplace of Eclipse. Then .metadata folder will be created in 'techniques' folder.
- On the 'Package Explorer' panel, Open context menu by clicking right mouse button.
- Select 'Import', Then a pop-up windows will be placed.
- Choose 'General > Projects from Folder or Archive' item and click 'Next' button.
- Designate project folder Locus.
-Then, the project will be loaded and be shown in the Package Explorer.

You can run Locus using command line: `java main.Main [config]`
It reads all the configurations from the [config] file.
Locus/config_example.txt is a example for config 

### SemanticCodeBERT

#### Dataset
he data we used to train and test is attached in link https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing.

Data_SenmanticCodeBERT.zip is the SenmanticCodeBERT. You need to download it and then extract it to SenmanticCodeBERT/data dir. Data_SenmanticCodeBERT.zip is converted from Data.zip, there is a convert tool in code.You can convert it by yourself

#### Experiment

you can read the README.md in SenmanticCodeBERT for train and evaluation

## CodeBERT

### Dataset

The same with KEPT

### Experiment


Before starting the experiment, there are some steps to follow, 

Step1: Install the python dependencies listed in the requirements.txt file within the directory CodeBERT.

Step2: Download the CodeBERT.zip file from https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing, then unzip it, and place the rapt.pt file in trace/main/model, and the remaining files in the /trace/unixCoder folder

Step3:Modify the DATA_SOURCE in trace/main/train_eval.sh to point to your data dir.

Step4: Modify the PROJECT_NAME in trace/main/train_eval.sh to point to the project you want to train and evaluate
#### Train & Evaluation

We provide a script that combines the training and evaluation processes. You can run it to reproduce our experiments.

The experimental results are in the trace/main/result folder
```bash
cd trace/main
bash train_eval.sh
```


## BERT

### Dataset

The same with KEPT

### Experiment


Before starting the experiment, there are some steps to follow, 

Step1: Install the python dependencies listed in the requirements.txt file within the directory BERT.

Step2: Download the BERT.zip file from https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing, then unzip it, and place the rapt.pt file in trace/main/model, and the remaining files in the /trace/unixCoder folder

Step3:Modify the DATA_SOURCE in trace/main/train_eval.sh to point to your data dir.

Step4: Modify the PROJECT_NAME in trace/main/train_eval.sh to point to the project you want to train and evaluate
#### Train & Evaluation

We provide a script that combines the training and evaluation processes. You can run it to reproduce our experiments.

The experimental results are in the trace/main/result folder
```bash
cd trace/main
bash train_eval.sh
```


## Kept_Pretrain

### Dataset

The data we used to train is attached in link https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing.

Data_Pretrain.zip is the KEPT pretrain dataset. You need to download it and then extract it to any location.
Then, modify the data_dir in rapt/run.sh to point to your data dir.

### Experiment

Before starting the experiment, there are some steps to follow, 

Step1: Install the python dependencies listed in the requirements.txt file within the directory KEPT.

Step2: Download the model data KEPT.zip file from https://drive.google.com/drive/folders/1ZQggquNR5vpFf3P7gPUPt7cZM8zy-L5Q?usp=sharing, then unzip it, and place all files to a directory.

Step3:modify the data_dir in rapt/run.sh to point to your data dir.

Step4: modify the code_bert in rapt/run.sh to point to your model dir

#### Train & Evaluate

```bash
cd rapt
bash run.sh
```
