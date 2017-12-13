## User-Agent-String-Parser

It parses the user agent string and predicts the family of the user agent. It converts the 
user agent string into a word count vector and then a trained classifier(SGDClassifier) is used to predict the family.
We use machine learning to predict families because they are defined classes and have less chance of change
Version is parsed from the string itself since it cannot be classified into different 
classes as it can be different for a new string. It will be difficult to create classes for versions

## How to use the code
python run.py --training path_of_training_data.txt --test path_of_test_data.txt --prediction-results path_of_output_data.txt

Example -
# python data_coding_challenge.py --training train.txt --test test.txt --prediction-results prediction.txt

## Modules-

-training() - training the classifier on training data
    arguments - training data (labeled tab separated file of user agent string, family and version)
    returns - trained classifier

-testing() - evaluating the classifier on testing data
    arguments - classifier, test data
    returns - user agent in test data, family of test data, predicted family of test user 
    agent strings, major versions of test data

-predict_versions() - parser to find major versions for each family
    arguments - user agent string from test data, predicted family of test data

-module of each family for returning version for each case

## Dependencies
-sklearn - classifier, feature engineering
-argparse - for creating argument parser for running python file
-numpy - for creating large vectors for big computations
