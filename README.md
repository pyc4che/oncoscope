# oncoscope ⚕️

[![en](https://img.shields.io/badge/lang-en-red.svg)](/README.md)
[![ua](https://img.shields.io/badge/lang-ua-green.svg)](/readmes/README.ua.md)

## Description 📄

This project was made for creating of Machine Learning Model for Tumor Detection on MRI Images. We are going to use `tensorflow` and `tensorflow.keras` libraries for training our model.

## Models 🤖

Using my notebooks I have generated two models, they are: `tumor_detection.keras` and `tumor_detection.h5`. Both are the same, though with different extensions.

## Installation ⚙️

Here is the detailed process of installation:

1. Clone the repository - `git clone https://github.com/pyc4che/oncoscope.git | cd oncoscope`
2. Create the Virtual Environment - `python -m venv .venv`
3. Activate the Virtual Environment - `source ./.venv/bin/activate`
4. Install required packages using `requirements.txt` file - `python -m pip install -r requirements.txt`

After all of this steps you are ready to launch the program, for this type: `jupyter notebook`. YOur default browser will be opened automatically.

## Machine Learning Model 🦾

The only notebook you see is `tumor_detection.ipynb` the main purpose of this notebook is the training, testing and saving of the model. It works with 100% accuracy !!!

Let's deepen into our Machine Learning process a bit.

### Packages 📦

For the Machine Learning process we have to read,  make our data preprocessed, train our model and save it. So for this all actions we need the variate of required packages, like:

1. Pandas - Read our Dataset;
2. Numpy - Linear Algebra actions;
3. Seaborn and Matplotlib - Data Visualization;
4. TensorFlow, TensorFlow.Keras, Scikit-Learn - Data Preprocessing and Model Training;
5. Some other libraries for different action;

### Data Processing 🧪

As dataset I am going to use this [kaggle dataset](https://www.kaggle.com/datasets/volodymyrpivoshenko/brain-mri-scan-images-tumor-detection)

1. Dataset Formation - Splitting data into topics and creation of Dataframe
2. Dataset Visualization - Plotting the Diagrams

As the result we have one diagram:

**Dataset Structure**
![dc](/output/dataset_structure.png)

### Preprocessing 🔨

Well, this is the most interesting part. Now, we are going to train our model and then save it. So let's move on.

This process divides into several steps:

1. Dataset Splitting - Preparing data for ML manipulations
2. Trining & Validation - Creating the Image Dataset
3. Modeling -  Training of the Model
4. Scoring the model - Analyzing the Model Results
5. Confusion Matrix - Post-Analysis, Visualizing Accuracy
6. Saving Model - Writing to a File

- *A confusion matrix in machine learning is a table that helps evaluate the performance of a classification algorithm.*

**Model Training**
![mt](/output/model_training.png)

**Confusion Matrix**
![cm](/output/confusion_matrix.png)

The models could be find by path: `/output/models/keras`.

## Thanks, Bye 👋🏻
