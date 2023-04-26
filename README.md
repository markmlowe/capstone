capstone
==============================

CLASS: DATA 4950 

AUTHOR: Mark M.Lowe

# Overview
------------------------------------------------------------------------------------------------------------------------------------------------------
- This repository contains a Logistic Regression model & an XGBoost Classifier model to predict a nurse's probability of attriting. 

    - The data was sourced from : https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare?resource=download
    
- For a more in depth understanding of this project, please look at the EDA inside of the jupyter notebook in the 'notebooks' folder

- Also, please read the summary (titled Nurse_Attrition_Project_Summary.pdf) located in the 'reports' folder


# Project Organization
------------------------------------------------------------------------------------------------------------------------------------------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project, parsed down to just contain important information
    ├── data
    │   ├── external       <- Data from third party source. All other folders are unused inside of 'data'
    │
    ├── docs               <- Contains AI/ML Bias-Fairness report
    │
    ├── notebooks          <- `01_mml_initial_data_exploration.ipynb` Contains EDA for Project.
    │
    ├── reports            <- PDF Summary of Project.
    │   └── figures        <- Generated graphics and figures used in PDF summary
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, 
    |                         use pip install -r requirements.txt in terminal
    │                         
    └── src                <- Source code for use in this project.
    │   ├── data           <- python script 'make_dataset.py' to create and split the data for training and testing. 
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions. Specifically 'train_models.py' and 'predict_models.py'
    │   │   ├── predict_models.py
    │   │   └── train_models.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations, 
    |                         specifically Confusion Matrices of individual models
    │       └── visualize.py
    
---------------------------------------------------------------------------------------------------------------------------------------------------------
# How-To
- Create virtual environment to install all dependencies : FOR HELP - https://realpython.com/python-virtual-environments-a-primer/

- Active the virtual environment and run 'pip install -r requirements.txt' in the terminal

- Once requirements are installed, run 'python run.py' to generate an iteration of the model (Both Logistic Regression & XGBoost Classifier)

    - The 'run.py' script calls to 'make_dataset.py' --> 'train_models.py' --> 'predict_models.py' --> 'visualize.py'
    - The 'run.py' script outputs the confusion matrix report  with all metrics and generates the 2x2 matrix as a pop-up the dev can save, if desired.
        - To continue the run from logistic regression to xgboost, the dev will need to exit out of the log_r pop-up graphic, allowing the script to run the xgboost model.

----------------------------------------------------------------------------------------------------------------------------------------------------------
# Contact Info:

- For any questions, please reach out to mml4w@mtmail.mtsu.edu 


