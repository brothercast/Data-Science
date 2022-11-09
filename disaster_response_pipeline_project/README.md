# Udacity Data Science Nanodegree
## Disaster Response Pipeline Project 
### Description
This project is part of Udacity Data Science Nanodegree program. 
The assignment is to create a Natural Language Processing (NLP) model for emergency responders to classify messages into categories.

The project is divided into three parts: 
**Extract-Transform-Load Pipeline (ETL)**
The ETL Pipeline is used to extract data from the source files (disaster_categories.csv and disaster_messages.csv), transform the data and load it into a SQLite database (DisasterResponse.db).

**Machine Learning Pipeline** 
The ML pipeline is used to train a model using GridSearchCV, evaluate its performance on test set and save the best model as pickle file for later use in web app. The trained classifier can be found under models/classifier.pkl folder of this repository.

**Web App: Disaster Response Project** 
A Flask web application that takes an input message and classifies it into categories. The web app also displays visualizations of the data in form of graphs and charts.

### Instructions:
1) Run the following commands in the project's root directory to set up your database and model.
    - Run the ETL pipeline to clean and tokenize the data and store it in a database:
    
    ```python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterResponseDB```
    
    - Run the Machine Learning pipeline that trains the classifier and saves the model as a Pickle file: 
    
    ```python3 train_classifier.py ../models/DisasterResponseDB models/classifier.pkl```

### Dependencies
- Python 3.5+ (I used Python 3.6.11)
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Web Scraping Libraries: BeautifulSoup, Requests/Regex
- Web App and Data Visualization: Flask, Plotly
- Model Loading and Saving Library: Pickle

### Other Files
app/templates/: Templates for the web app.
app/run.py: Start the Python server for the web app and prepare visualizations.

### Author 
[Tone Pettit](https://github.com/brothercast)
