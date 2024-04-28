
## Core code for paper  "Redar: Recommendation-guided Active Trapping for Ransomware"

![framework.png](images%2Fframework.png)
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

## 3. Project files
For copyright reasons the presented project only contains the code of Decoy File Generation 
component, which contains eight files show the core part of the codes. Here is an overview 
of the main features of each file below:
### 3.1 main.py
   Here is the main entry point for program execution, including defining model training parameters, 
   processing the entry of embedded vector data collected from user file attributes and ransomware samples, 
   the entry for training Redar to learn interaction relationship models, 
   the entry for training Redar to learn attribute weights, 
    the entry for decoy file generation functionality, 
   and the entry for testing the efficiency of decoy file detection.
   
### 3.2 data_loader.py
Here, we obtain the initial embedding vectors of ransomware samples and user file attributes from the dataset, 
forming them into two graph representations: the user file attribute graph and the ransomware sample graph.

### 3.3 train.py
Here, the interaction information learning is conducted between the user file attribute graph generated in 
data_loader.py and the ransomware sample graph. Three types of interaction information are learned here: 
internal interaction information of user file attributes, aimed at enabling Redar to learn correlation information
between file attributes; internal interaction information of ransomware samples, aimed at enabling Redar to learn 
correlation information between ransomware samples; and interaction information between ransomware samples and file
attributes, aimed at enabling Redar to learn preference information when ransomware samples encrypt files. The final
ransomware sample embedding representation and file attribute embedding representation obtained in this step contain
new embedding vectors that include the above three types of information.

### 3.3 model.py
This section specifically defines the training process in train.py, which continuously adjusts the parameters in GNN to 
ensure that the resulting embedding vectors for ransomware samples and user file attributes contain sufficient 
representations of the three types of interaction information mentioned in Section 3.3.

### 3.4 weight_method.py
This section defines a linear model to learn the weights of different types of file attributes in determining the 
attributes that decoy files should possess, i.e., the overall attention of ransomware samples to each type of user file 
attribute. The input to this section is the embedding vectors of ransomware samples obtained in Section 3.3 and the 
embedding vectors of user file attributes. The output is a list of weights for each type of user file attribute.