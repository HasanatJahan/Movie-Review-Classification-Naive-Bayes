"""
Name: Hasanat Jahan
Homework 2 Naive Bayes Implementation on Movie Review 
"""

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


"""
Function used to iterate through the nested dictionary 
"""
def get_all_words(dictionary, output_dict):
    for key, value in dictionary.items():
        if type(value) is dict:
            get_all_words(value, output_dict)
        else:
            if key not in output_dict:
                output_dict[key] = 1
            else:
                output_dict[key] += 1 



"""
Function used to build the model parameters that would store the model parameters in a file
"""
def build_parameter_file(training_file, parameter_file):
    model_parameter_dict = dict()

    # open training file and read through it 
    f = open(training_file)
    training_data = json.load(f)
    f.close()

    # get the number of documents 
    num_document = len(training_data)

    label_dict = dict()
    all_training_word_dict = dict()
    # iterate through the dictionary to draw out the labels to create prior probability 
    for i in training_data:
        for j in i:
            if j not in label_dict:
                label_dict[j] = 1
            else:
                label_dict[j] += 1  

            get_all_words(i, all_training_word_dict)    

    print(all_training_word_dict)

    # now to calculate prior probability of each label 
    for label in label_dict:
        prior_prob = label_dict[label]/num_document
        # append it to the dictionary 
        prob_label_name = "P(" +  label   + ")"
        # print(prob_label_name)
        model_parameter_dict[prob_label_name] = prior_prob
    
    # # now to get all words so as find the size of the vocabulary
    # get_all_words(training_data, all_training_word_dict)
    # print(all_training_word_dict)
    
    


def run_naive_bayes(training_file, testing_file, parameter_file, output_prediction_file):
    print()


build_parameter_file(training_file, parameter_file)
run_naive_bayes(training_file, testing_file, parameter_file, output_file)

