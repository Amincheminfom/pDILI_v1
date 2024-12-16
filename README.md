## What is this pDILI_v1?

<img src="https://github.com/Amincheminfom/pDILI_v1/blob/main/pDILI_logo.jpg" alt="pDILI Logo" width="250" align="right"/>

**pDILI_v1** is an online tool hosted on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SXPN1QfGngGKRZQgvWPytrfZv8q1TX3N#scrollTo=4SVqxdmO0MQM). This tool also provides Graphical User Interface (GUI) which can be run through Anaconda environment (in Windows). This tool predict the association of drug-induced liver injury of a small molecule (1 = RISKy, 0 = Non-RISKy) and also visualize the molecule.



pDILI_v1 stands for **p**redictor of **D**rug-**I**nduced **L**iver **I**njury. 

This Google Colab notebook is a supplementary material of the paper "pDILI_v1: A Machine Learning-Based Tool for Predicting Drug-Induced Liver Injury (DILI) Integrating Chemical Space Analysis and Molecular Fingerprints" (manuscript under preparation).

---
# For Google Colab Users:
Please follow these three steps before running this notebook.

1: Download the two csv files provided herewith (named '1_train_pDILI.csv' and '2_test_pDILI.csv') and create a folder named "pDILI_v1". Move these two csv files in to the folder **pDILI_v1**.

or Download the folder named "pDILI_v1" [Directly Download](https://drive.google.com/drive/u/1/folders/1r1NZOxiNmtwSyYogbYTXuDd7ymN9I_Sc).

2: Upload this folder (**pDILI_v1**) in your Google Drive. Copy this path. Make sure that ''1_train_pDILI.csv' and '2_test_pDILI.csv' are present in that folder **pDILI_v1**.

3: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SXPN1QfGngGKRZQgvWPytrfZv8q1TX3N#scrollTo=4SVqxdmO0MQM) and execute it to predict the DILI RISK property of the query molecule as well as visualize the Applicability domain (AD).

---
Example Smiles:

(a) Sorbitol: C(C(C(C(C(CO)O)O)O)O)O

(b) Almotriptan: CN(C)CCC1=CNC2=C1C=C(C=C2)CS(=O)(=O)N3CCCC3

(c) Imatinib: Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1

---
# For Graphical User Interface (GUI) Users:

1: Installation of Anaconda
[Downalod Anaconda](https://www.anaconda.com/download)  & Install it.

If you already have Anaconda in your system then skip the Installation.

2: Open Anaconda Prompt

3: Then activate environment (for example: pDILI_v1). You can copy and run this command directly in your terminal:

   ```bash
   conda create -n pDILI_v1 python=3.9  #for the first time only
   ```
You can copy and run this command directly in your terminal:

   ```bash
   conda activate pDILI_v1
   ```
3(for the first time only). Then install the required packages (one time only)
You can copy and run this command directly in your terminal:

   ```bash
   conda install -c anaconda tk #for the first time only
   ```
   ```bash
   conda install -c conda-forge rdkit pandas scikit-learn pillow #for the first time only
   ```
#or
   ```bash
   pip install rdkit pandas scikit-learn pillow #for the first time only
   ```
   ```bash
   pip install matplotlib numpy mordred #for the first time only
   ```
4. Go to your working directory

  ```bash
cd yourpath #for example D:\DILI_Amin\pDILI_v1
   ```
Ensure that the training, test set, the pDILI_logo and the pDILI_v1.py should be present in the your working directory (pDILI_v1).

Then run the command
   ```bash
   python pDILI_v1.py
   ```
---
Bugs: If you encounter any bugs, please report the issue to [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com).

