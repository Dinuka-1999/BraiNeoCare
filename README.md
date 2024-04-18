# Deep Learning for Real-Time Neonatal Seizure Detection with Reduced Montage

Notice: This project was originally carried out in Ubuntu 22.04.4 LTS

## Problem background

* Neonatal seizures are epileptic seizures that occur in infants during the first four weeks after birth. This duration is the most vulnerable time in a lifetime to develop seizures capable of causing significant harm to the developing brain, necessitating prompt diagnosis followed by treatment.

* Detecting neonatal seizures is particularly challenging because they often manifest subtly and can be mistaken for normal physiological behaviors. Therefore, having an objective monitoring method is critical.

* Hence we introduce a new efficient yet reliable and accurate AI model to detect neonatal seizures with reduced electrode montage. Here we have used only 9 electrodes and 12 EEG channels namely ["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"].

## Run the program

### 1) Create a conda environment

If you haven't installed Conda on your PC, first please install Conda by referring to the following link. https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Once you successfully installed Conda on your PC, you can start by creating a Conda environment for the project. Let's say the name of the environment is BraineoCare
```
conda create --name BraineoCare
```

Next, activate the conda environment by calling the command,

```
conda activate BraineoCare
```
### 2) Clone the GitHub repository

Next, clone the GitHub repository by running the following command in your terminal after navigating to a preferred location. Or else you can download the zip from the GitHub repo.

```
git clone https://github.com/internTrio/peopleCounter
```

### 3) Install required libraries
The next step is to install the required libraries. Before that, make sure you have activated the Conda environment we created for the project. If not, activate it using the command we discussed before. Also, navigate inside to the git repo,
```
cd peopleCounter
```
Once you are in the correct directory and activated the Conda environment, watch [This video](https://www.youtube.com/watch?v=dZh_ps8gKgs) to install the Tensorflow object detection API.
After that run the code below to install the required libraries.

```
pip install -r requirements.txt
```


