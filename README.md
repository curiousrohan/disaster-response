# Disaster Response Pipeline 
The goal of this project is to accurately classify which responses needs attention during emergency situation.
A web dashboard is made to solve this probloem.

## Content
- Data
  - process_data.py: Reads in the dataset, cleans and processes it and stores it in a SQL database.
  - disaster_categories.csv(Dataset Containing 36 types of categories) and disaster_messages.csv(Messages during a disaster)
  - DisasterResponse.db: Output file from process_data.py. Stores the processed data in a database.
- Models
  - train_classifier.py: Loads the DisasterResponse.db and does text cleaning before feeding the output to a machine learning pipiline using Random Forests. GridSearchCV is used to tune the hyperparametets. Model is saved as a pickel file.
  - classifier.pkl: Saved Machine Learning model
- App
  - run.py: Flask app and the GUI used to predict results and display them. Uses classifier.pkl as model.
  - templates: Folder containing the html templates

## Usage:

> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

> python train_classifier.py ../data/DisasterResponse.db classifier.pkl

> python run.py

> Go to http://0.0.0.0:3001 to visualize the Web Ap
## About
The data was provided by Figure Eight as a part of the Udacity Data Scientist Nanodegree programme.
