# Disaster Response Pipeline Project

### Project Overview
In this course, we've learned and built on your data engineering skills to expand your opportunities and potential as a data scientist. In this project, we'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
In the Project, we use a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. 
Our project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off our software skills, including our ability to create basic data pipelines and write clean, organized code!

### Project Components
There are three components we'll need to complete for this project. 

#### 1. ETL Pipeline
In a Python script, process_data.py, we write a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data 
- Stores it in a SQLite database

#### 2. ML Pipeline
In a Python script, train_classifier.py, we write a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### 3. Flask Web App
The flask web app uses the knowledge of flask, html, css and javascript and provides an interface for the user to classify the messages in different categories. The web application also provides the following visualizatios for the project using plotly:
- Distribution of Message Genres
- Distribution of Categories

### Instructions to run the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
