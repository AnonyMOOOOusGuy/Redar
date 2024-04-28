
## Core code for paper  "Redar: Recommendation-guided Active Trapping for Ransomware"


## 1. Introduction to Project 
This project is the core algorithmic code of component "Decoy File Generator" in paper "Recommendation-guided Active 
Trapping for Ransomware" It mainly includes the following steps:
1. Transform the interaction records between training ransomware samples and user files into embedding vectors.
2. Use GNN to learn interaction information (inner and cross) of embedding vectors, creating new representations.
3. Calculate different attribute classes scoring weights using the embedding vectors of ransomware samples and user files.
4. Based on the weights of each file attribute class and the scores of each ransomware for each file attribute, obtain a candidate set of decoy file attributes.
5. Apply the same operations to the paths of the target file system and select the candidate file path set that all ransomware pays attention to overall.
6. According to the candidate file paths with the set of decoy file attributes, select the optimal decoy files.
7. Test the effectiveness of the selected decoy files under different test conditions.

## 2. How to run the project
To make the project work well ,you may need third party Python dependencies as below:
* Python 3.8
* Pytorch 1.13.1
* torch_geometric (capable for Python 3.8)
* numpy
* pandas
* sklearn

