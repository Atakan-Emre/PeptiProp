# GEPPRI
The interactions between proteins and peptides are one of the most crucial biological interactions and are implicated in abnormal cellular behaviors leading to diseases such as cancer. Thus, knowledge of these interactions provides invaluable insights into all cellular processes such as DNA repair, replication, gene expression and metabolism, and drug discovery to treat many diseases including cancer. 
In this regard, we introduce a stack ensemble deep learning model, GEPPRI, that incorporates various features obtained from protein sequences and structures. The proposed framework relies on three pipelines: pre-processing, processing, and post-processing. In other words, the combination of employed close neighbor correlation coefficient (CNCC), Half-sphere exposure group (HSE), local backbone angles (LBA), physicochemical properties (PP), amino acid occurrence (AAO), and a type of stack ensemble deep learning model resulted in superior performance compared to the sequence-based and structured-based state-of-the-art methods using the two commonly datasets.

# GEPPRI
****Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.****

Please cite the relevant publication if you will use this study.

***Citation: S. Shafiee, A. Fathi and G. Taherzadeh, Protein-peptide interaction region residues prediction using generative sampling technique and ensemble deep learning-based models.***

# Data Files

****The two commonly used datasets are as follows:****

Train1

Train2

Test1

Test2

****Dataset1_GEPPRI include Train1, and Test1, and Dataset2_GEPPRI contain Train2, and Test2, respectively.****

# Guide

In FASTA format the line before the protein sequence, called the FASTA definition line, must begin with a carat (">"), followed by a unique SeqID (sequence identifier). The SeqID must be unique for each protein sequence and should not contain any spaces. For a fair comparison, Dataset 1 (including Test1, and Train1), and Dataset 2 (including Test2, and Train2) for model training and testing, respectively are employed.

# Code Files

****Our source code includes the following files. This source code relies on three pipelines as follows:****

**The pre-processing pipeline comprises the following files:**

Feature groups

Pre-process procedure

PreGAN1

PreGAN2

PreGAN3

DGAN

**The processing pipeline includes the following files:**

DCNN 

DLeNet

DResNet 

**The post-processing pipeline contains the following files:**

Stacked ensemble deep learning framework 

# Guide

The result obtained in our study can be replicated by executing the Dataset1_GEPPRI for Dataset1, and Dataset2_GEPPRI for Dataset2. To achieve the result of GEPPRI on dataset1, all the above files should be provided in one folder and run the pre-processing pipeline, processing pipeline, and post-processing pipeline respectively in an operating system.

# System requirement

numpy 1.21.2

pandas 1.1.3

torch 1.8.0

biopython 1.79

cv2 ( interface in OpenCV versions is named as cv )

python 3.7.9

math1.2.7

scipy 1.2.9

cPickle 1.2.8

pickle 1.2.7

# Contact

For further details or questions, it is possible to communicate via email (shafiee.shima@razi.ac.ir)

% Best Regards % 
