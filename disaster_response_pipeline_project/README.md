# Udacity Data Science Nanodegree
## Disaster Response Pipeline Project 
### Description
This project is part of Udacity Data Science Nanodegree program. 
The assignment is to create a Natural Language Processing (NLP) model for emergency responders to classify messages into categories.

The project is divided into three parts: 
#### Extract-Transform-Load Pipeline (ETL)
The ETL Pipeline is used to extract data from the source files (disaster_categories.csv and disaster_messages.csv), transform the data and load it into a SQLite database (DisasterResponse.db).

#### Machine Learning Pipeline 
The ML pipeline is used to train a model using GridSearchCV, evaluate its performance on test set and save the best model as pickle file for later use in web app. The trained classifier can be found under models/classifier.pkl folder of this repository.

#### Web App: Disaster Response Project
A Flask web application that takes an input message and classifies it into categories. The web app also displays visualizations of the data in form of graphs and charts.

<img src="https://github.com/brothercast/Data-Science/blob/master/disaster_response_pipeline_project/img/DizRespApp2.png" width="400">
<img src="https://github.com/brothercast/Data-Science/blob/master/disaster_response_pipeline_project/img/DizRespApp.png width="400">

### Instructions:
1) Run the following commands in the project's root directory to set up your database and model.
    - Run the ETL pipeline to clean and tokenize the data and store it in a database:
    
    ```python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterResponseDB```
    
    - Run the Machine Learning pipeline that trains the classifier and saves the model as a Pickle file: 
    
    ```python3 train_classifier.py ../models/DisasterResponseDB models/classifier.pkl```

### Dependencies
- Python 3.5+ (Python 3.6.11 used)
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Web Scraping Libraries: BeautifulSoup, Requests/Regex
- Web App and Data Visualization: Flask, Plotly
- Model Loading and Saving Library: Pickle

### Other Files
- **app/templates/:** HTML templates for the web app.
- **app/run.py:** Starts the Python server for the web app and prepares visualizations.
- **data/process_data.py:** Extracts, transforms and loads the data into a SQLite database (DisasterResponseDB). 
- **models/train_classifier.py:** Builds an NLP model using GridSearchCV to classify messages in categories based on training dataset stored in DisasterResponseDB. The trained classifer is saved as pickle file for later use by web app.
- **data/disaster_categories.csv:** Dataset containing message categories (36 in total). 
- **data/disaster_messages.csv:** Dataset containing messages and their corresponding category labels from disaster_categories dataset.

### Author 
[Tone Pettit](https://github.com/brothercast)
