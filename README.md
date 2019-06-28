# HAR

Code for WWW2019 paper "A Hierarchical Attention Retrieval Model for Healthcare Question Answering" (http://dmkd.cs.vt.edu/papers/WWW19.pdf)

** The data preprocessing and training code are from MatchZoo branch 1.0. Please refer to the original repo for more details: https://github.com/NTMC-Community/MatchZoo **

## HealthQA Dataset

Please email author(mingzhu@vt.edu) for further access.

## Data Preprocessing

You should treat your data format as 'sample.txt', formatted as 'label \t query\t document_txt'.In detail, label donotes the relation between query and document, 1 means the query is related to the document, otherwise it does not matter. The words in query and documents are separated by white space. 

Name the training, testing, and validation data as "sample-mz-train.txt", "sample-mz-test.txt", "sample-mz-dev.txt". Put the data files under /data/sample folder.

Copy all the scripts in /pinfo to /sample. Run run_data.sh to process the data.

## Training

The HAR model is named *mymodel* in the repo. To run it, first create a folder under /examples. Name the folder /sample. Create a config file and put in the folder. You can copy the config file of *mymodel* from the /pinfo folder. Modify the config file to fit your needs.

To train:
python matchzoo/main.py --phase train --model_file examples/pinfo/config/mymodel_pinfo.config

To predict:
python matchzoo/main.py --phase predict --model_file examples/pinfo/config/mymodel_pinfo.config




