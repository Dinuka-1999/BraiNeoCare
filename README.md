# Explainable Deep Learning for Real-Time Neonatal Seizure Detection with Reduced Montage

Notice: This project was originally carried out in Ubuntu 22.04.4 LTS

## Problem Background

* Neonatal seizures are epileptic seizures that occur in infants during the first four weeks after birth. This duration is the most vulnerable time in a lifetime to develop seizures capable of causing significant harm to the developing brain, necessitating prompt diagnosis followed by treatment.

* Detecting neonatal seizures is particularly challenging because they often manifest subtly and can be mistaken for normal physiological behaviors. Therefore, having an objective monitoring method is critical.

* Hence we introduce a new efficient yet reliable and accurate AI model to detect neonatal seizures with reduced electrode montage. Here we have used only 9 electrodes and 12 EEG channels namely ["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"].

## Getting Started

### Proposed Model Architecture 
![Full_model](https://github.com/Dinuka-1999/BraiNeoCare/assets/81279517/70ea85b7-8bc5-4f42-92de-e7dbfa5867a5)


### 1) Create a Conda Environment

If you haven't installed Conda on your PC, first please install Conda by referring to the following link. https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Once you successfully installed Conda on your PC, you can start by creating a Conda environment for the project. Let's say the name of the environment is BraiNeoCare
```
conda create --name BraiNeoCare
```

Next, activate the conda environment by calling the command,

```
conda activate BraiNeoCare
```
### 2) Clone the GitHub Repository

Next, clone the GitHub repository by running the following command in your terminal after navigating to a preferred location. Or else you can download the zip from the GitHub repo.

```
git clone https://github.com/Dinuka-1999/BraiNeoCare
```

### 3) Install Required Libraries
The next step is to install the required libraries. Before that, make sure you have activated the Conda environment we created for the project. If not, activate it using the command we discussed before. Also, navigate inside to the git repo,
```
cd BraiNeoCare
```
Once you are in the correct directory and activated the Conda environment, run the code below to install the required libraries.

```
pip install -r requirements.txt
```

### 4)Dataset
* To run the files, you need to download the publicly available Zenodo Neonatal EEG dataset published by Helsinki University. You can find the dataset [here](https://zenodo.org/records/4940267). Please make sure to download version 4 of the dataset. It is recommended to create a folder named "Datasets" and download the dataset into that folder.
```
Datasets\
|------- Zenodo_eeg\
|        |--------- annotations_2017.mat
|        |--------- eeg1.edf
|        |           :
|        |--------- eeg79.edf
|------- processed_data\
         |--------- traindata.npy
         |--------- trainlabels.npy
         |--------- testdata.npy
         |--------- testlabels.npy
```

After executing the Read_Data.py file, you can save the pre-processed data required for training and testing the machine learning model. Run the code below to execute the file.
```
python Read_Data.py
```
## Training the Models

There are three files if you want to train a model.
```
|----- Model.py
|----- cross_validation.py
|----- SSL.py
```
To train the model, you have the option to run any of these files. However, we strongly suggest running either the first or second file, instead of the third one. The first file will divide the dataset into train and test datasets at a ratio of 4:1, while the second file will perform **10-fold cross-validation**. Please note that if you use the SSL.py file, you will receive a pre-trained model. However, you will need to download an unlabeled neonatal EEG dataset to use it. 

## Model Interpretability and Inferencing 

### Model Interpretability

The algorithm presented here aims to facilitate the comprehension of the concept of model interpretability. It can help users gain a better understanding of how a model makes predictions and the factors that influence them.

![Screenshot 2024-04-18 154351](https://github.com/Dinuka-1999/BraiNeoCare/assets/81279517/fe5a342a-4c57-405e-a08b-86b0bee9ce86)

Run the file **XAI.py** to see the outputs as follows.
![EX_AI](https://github.com/Dinuka-1999/BraiNeoCare/assets/81279517/77a56722-2b60-4d9c-bada-91cb724912b9)



## GUI

For real-time seizure detection using a trained model, run the **gui_mp.py** file. You may need to add relevant paths to the trained model and other files. Before running this file please read the following carefully which describes this Python script.

* First of all note that, we finally applied this trained deep learning model to data we acquired from a new EEG headset that we developed. Therefore, if you run the script without necessary changes you will get errors.
*  As the first step go to the get_data() function and remove or comment lines from 285 to 334 and from 341 to 365 and uncomment the lines from 367 to 373 and 335th line. Also, read the eeg to an array``` data2=np.load("bc_recordings.npy")```
*  Secondly go to the filter_data() function and remove or comment the line 197.
*  These are the necessary changes you need to make to run the script. After that, a window will open and you will find several buttons on the left side of the window. You are advised to press only the **start** button to see the signal in the loaded data2 file as a live run. Then you may press the **stop** button to stop the data reading process and **Run ML Model** to load the trained deep learning model to detect seizures in real time. After pushing this button you will see the interpretability as well.

The following short video shows an example of real-time seizure detection. The green color represents normal EEG signal and the red color represents seizures. We used these two colors for clarity. In the second half of the video, you can see the offline yet efficient and accurate artifact removal (Not related to the main deep learning algorithm. But this was also implemented with a deep learning model)

https://github.com/Dinuka-1999/BraiNeoCare/assets/81279517/fac5e828-8e20-4523-a7a5-88d281f7a33a

## Future works 

As of now, we couldn't get the maximum benefit out of the Self-supervised learning model. Therefore in our future work, we will develop version two of this proposed model.

## Cite
BibTeX of the original publication in the proceedings of IEEE International Conference on Systems, Man, and Cybernetics 2024 will soon replace this 
```
@article{udayantha2024using,
  title={Using Explainable AI for EEG-based Reduced Montage Neonatal Seizure Detection},
  author={Udayantha, Dinuka Sandun and Weerasinghe, Kavindu and Wickramasinghe, Nima and Abeyratne, Akila and Wickremasinghe, Kithmin and Wanigasinghe, Jithangi and De Silva, Anjula and Edussooriya, Chamira US},
  journal={arXiv preprint arXiv:2406.16908},
  year={2024}
}
```
