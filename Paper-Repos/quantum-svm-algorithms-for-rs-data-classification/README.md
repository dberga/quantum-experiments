# Quantum Support Vector Machine Algorithms for Remote Sensing Data Classification

## General information

🗃 This repository contains Python functions and processing pipelines documented in Jupyter notebook for pixel-wise binary classification of remote sensing multispectral images with the D-Wave Advantage quantum annealer.

### Current publication 

More information can be found in the conference paper connected to this repository

📜 Amer Delilbasic, Gabriele Cavallaro, Madita Willsch, Farid Melgani, Morris Riedel and Kristel Michielsen, “Quantum Support Vector Machine Algorithms for Remote Sensing Data Classification”, in Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2021 (accepted). 

Recent developments in Quantum Computing (QC) have paved the way for an enhancement of computing capabilities. Quantum Machine Learning (QML) aims at developing Machine Learning (ML) models specifically designed for quantum computers. The availability of the first quantum processors enabled further research, in particular the exploration of possible practical applications of QML algorithms. In this work, quantum formulations of the Support Vector Machine (SVM) are presented. Then, their implementation using existing quantum technologies is discussed and Remote Sensing (RS) image classification is considered for evaluation.

### Previous publications 

📃 D. Willsch, M. Willsch, H. De Raedt, and K. Michielsen, “Support Vector Machines on the D-Wave Quantum Annealer” in Computer Physics Communications, vol. 248, 2020, https://doi.org/10.1016/j.cpc.2019.107006 

📃 G. Cavallaro, D. Willsch, M. Willsch, K. Michielsen, and M. Riedel, “Approaching Remote Sensing Image Classification with Ensembles of Support Vector Machines on the D-Wave Quantum Annealer,” in Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS), pp. 1973-1976, 2020, https://doi.org/10.1109/IGARSS39084.2020.9323544  

### D-Wave Leap 

👌 Everyone can make a free account to run on the D-Wave Advantage quantum annealer: 

- Sign up on D-Wave Leap through 👉 https://www.dwavesys.com/take-leap

- Install Ocean Software with 'pip install dwave-ocean-sdk' 👉 https://docs.ocean.dwavesys.com/en/latest/overview/install.html

- Configure the D-Wave System as a Solver with 'dwave config create' 👉 https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html

- Check the available computing time for your account on the D-Wave Leap Dashboard 👉 https://cloud.dwavesys.com/leap/login/

### IBM Quantum Experience

👌 Everyone can make a free account to run on IBM quantum machines and simulators: 

- Sign up and login on IBM Quantum Experience through 👉 https://quantum-computing.ibm.com/login

- Install Qiskit with 'pip install qiskit' 👉 https://qiskit.org/documentation/getting_started.html

- In a notebook, configure the provider using your personal token 👉  https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq

## Experiments

### Praparation of the binary classification problem

The binary classification problem is constructed from the SemCity Toulouse multispectral benchmark data set, that is publicly available 👉 https://doi.org/10.5194/isprs-annals-V-5-2020-109-2020. More information about the dataset can be found in the publication below 


R. Roscher, M. Volpi, C. Mallet, L. Drees, and J. D. Wegner, “Semcity toulouse: a benchmark for building instance segmentation in satellite images,” ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. V-5-2020, p. 109–116, 2020.

The processing workflow to build the classification problem is in this Jupyter Notebook 👉 Build_train_and_test_sets.ipynb

The specific data that we used for the experiments in the paper are in the folder 👉 experiments/data

Training set:
- X_train_tile_4_tiny.npy
- Y_train_tile_4_tiny.npy

Test set:
- X_test_tile_8_subregion.npy
- Y_test_tile_8_subregion.npy

### Classification with classical SVM (Scikit-Learn)

Follow the instructions of the Jupyter Notebook 👉 experiments/Classic_SVM/Classic_SVM.ipynb

### Classification with QA-based QSVM (D-Wave QA)

Follow the instructions of the Jupyter Notebook 👉 experiments/QA_SVM/QA_SVM.ipynb

### Classification with Circuit-based QSVM (IBM Quantum Experience)

Follow the instructions of the Jupyter Notebook 👉 experiments/Circuit_SVM/Circuit_SVM.ipynb

## Support

📬 For any problem, feel free to contact me at g.cavallaro@fz-juelich.de 

## Additional Bibliography and Sources


P. Rebentrost, M. Mohseni, and S. Lloyd, “Quantum support vector machine for big data classification,” Physical Review Letters, Sep 2014

D. Anguita, S. Ridella, F. Rivieccio, R. Zunino, "Quantum optimization for training support vector machines", Neural Networks, 2003

"Implementing QSVM Machine Learning Method on IBM's Quantum Computers", Quantum Computing UK, 2020

X. Zhu, J. Xiong and Q. Liang, "Fault Diagnosis of Rotation Machinery Based on Support Vector Machine Optimized by Quantum Genetic Algorithm," in IEEE Access, vol. 6, pp. 33583-33588, 2018

A. K. Bishwas, A. Mani and V. Palade, "Big data classification with quantum multiclass SVM and quantum one-against-all approach," 2016 2nd International Conference on Contemporary Computing and Informatics (IC3I), Noida, pp. 875-880, 2016

D. Uke, K. K. Soni and A. Rasool, "Quantum based Support Vector Machine Identical to Classical Model," 2020 11th International Conference on Computing, Communication and Networking Technologies (ICCCNT), Kharagpur, India, pp. 1-6, 2020.
