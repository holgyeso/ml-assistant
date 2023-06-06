# ML Assistant
The aim of the project is to build a Django-based Python back-end of an application with the following functionalities:
* be able to upload a CSV file
* EDA & cleaning:
    * see the number of rows and columns
    * display all column names and their dtype from the uploaded dataset
    * inspect first and last N rows
    * display statistics on the columns (mean, min, max, std, etc.)
    * delete missing data if requested
* Machine Learning:
    * choose from the column features that will be provided to an AI algorithm
    * transform/normalize the selected columns as specified in the user input
    * have the possibility to choose a regression, classification or clustering algorithm to train
    * make a form dynamically based on the features and selected algorithm, where users can input some data and use the pre-trained model to make predictions

## How to get it work?

Working in a terminal:
1. clone this repo with `git clone https://github.com/holgyeso/ml-assistant.git`
2. enter into the _ml-assistant_ directory with `cd ml-assistant`
3. start the Django web application with the python manage.py runserver command
4. navigate in your browser to http://localhost:8000/, where the application should appear