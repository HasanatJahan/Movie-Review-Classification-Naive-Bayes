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

import json

# Here we would have input from the command line but for how we have placeholders 


# trying with the small file 
training_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/train_feature_vectors.json'
testing_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/test_feature_vectors.json'
parameter_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/movie_review_small.NB'
output_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/output.txt'


def build_parameter_file(training_file, parameter_file):
    # open training file and read through it 
    f = open(training_file)
    training_data = json.load(f)
    f.close()

    # get the number of documents 
    # print(training_data)
    num_document = len(training_data)

    label_dict = dict()

    # iterate through the dictionary to draw out the labels to create prior probability 
    for i in training_data:
        for j in i:
            if j not in label_dict:
                label_dict[j] = 1
            else:
                label_dict[j] += 1  
    


def run_naive_bayes(training_file, testing_file, parameter_file, output_prediction_file):
    print("running")


build_parameter_file(training_file, parameter_file)
run_naive_bayes(training_file, testing_file, parameter_file, output_file)

