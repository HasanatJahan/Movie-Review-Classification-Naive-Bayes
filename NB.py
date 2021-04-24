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
import os 

# Here we would have input from the command line but for how we have placeholders 

# trying with the small file 
training_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/train_feature_vectors.json'
testing_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/test_feature_vectors.json'
parameter_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/movie_review_small.NB'
output_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/output.txt'


"""
Function used to build the model parameters that would store the model parameters in a file
"""
def build_parameter_file(training_file, parameter_file):
    model_parameter_dict = dict()

    # open training file and read through it 
    f = open(training_file)
    training_data = json.load(f)
    f.close()

    # NOTE USE THE ORIGINAL TRAINING DATA FOR EASIER DOCUMENT COUNTING 
    # get the number of documents 
    num_document = len(training_data)
    label_dict = dict()
    class_BOW = dict()
    num_of_words_in_class = dict()
    vocab_dict = dict()

    # iterate through the dictionary to draw out the labels to create prior probability 
    for i in training_data:
        for key, value in i.items():
            if key not in label_dict:
                label_dict[key] = 1 
                # declare an empty dict here 
                class_BOW[key] = {}               
            else:
                label_dict[key] += 1 
            
            
            for word, word_count in value.items():
                if key in num_of_words_in_class:
                    num_of_words_in_class[key] += word_count
                else:
                    num_of_words_in_class[key] = word_count
                
                if word not in vocab_dict:
                    vocab_dict[word] = word_count
                else:
                    vocab_dict[word] += word_count

                if word not in class_BOW[key]:
                    class_BOW[key][word] = word_count
                else:
                    class_BOW[key][word] += word_count


    num_vocab = len(vocab_dict)

    # print(f"This is num vocab {num_vocab}")
    # print(f"classBOW {class_BOW}")
    # print(f"Number of words in class {num_of_words_in_class}")
    # print(f"Label dict {label_dict}")
    # print(f"vocab dict {vocab_dict}")

    # now to calculate prior probability of each label and add it to dictionary 
    for label in label_dict:
        prior_prob = label_dict[label]/num_document
        # append it to the dictionary 
        prob_label_name = "P(" +  label   + ")"
        # print(prob_label_name)
        model_parameter_dict[prob_label_name] = prior_prob


    # print(f"Model parameters {model_parameter_dict}")
    # now to create the BOW parameters with add one smoothing 
    for label, word_dict in class_BOW.items():
        for word, word_count in word_dict.items():
            calculated_prob = (class_BOW[label][word] + 1)/(num_of_words_in_class[label] + num_vocab)
            prob_label_name = "P(" +  word   + "|" + label + ")"
            # print(prob_label_name, calculated_prob)
            model_parameter_dict[prob_label_name] = calculated_prob
    
    # print(model_parameter_dict)
    # now to write to the parameter file 
    if not os.path.exists(parameter_file):
        try:
            os.makedirs(parameter_file)
        except Exception as e:
            print(e)
            raise

    param_filename = os.path.split(parameter_file)[1]
    param_path = os.path.split(param_filename)
    param_path = os.path.dirname(os.path.normpath(parameter_file))

    print(f"param path {param_path}")
    print(f"param filename {param_filename}")
    with open(os.path.join(param_path, param_filename), 'w') as outfile:
        outfile.write(json.dumps(model_parameter_dict, indent=4))
    


def run_naive_bayes(training_file, testing_file, parameter_file, output_prediction_file):
    print()


build_parameter_file(training_file, parameter_file)
run_naive_bayes(training_file, testing_file, parameter_file, output_file)

