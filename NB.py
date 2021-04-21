import json

"""
should take the following parameters: 
- the training file 
- the test file 
- the file where the parameters of the resulting model will be saved 
- the output file where you will write predictions made by the classifier 
on the test data (one example per line). 
The last line in the output file should list the overall accuracy of 
the classifier on the test data. 
The training and the test files should have the following format: 
one example per line; each line corresponds to an example; 
first column is the label, 
and the other columns are feature values.
"""

# Here we would have input from the command line but for how we have placeholders 


# trying with the small file 
training_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors'


def run_naive_bayes(training_file, testing_file, parameter_file, output_prediction_file):
    print("running")