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
import math
# Here we would have input from the command line but for how we have placeholders 

# for the main movie review
# training_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
# testing_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
# parameter_file = 'movie-review-HW2/movie-review-BOW.NB'
# output_file = 'movie-review-HW2/output.txt'


# for experimental review with additional features 
# training_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
# testing_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
# parameter_file = 'movie-review-HW2/movie-review-BOW-experiment.NB'
# output_file = 'movie-review-HW2/experiment-output.txt'


# for small movie review file 
# training_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/train_feature_vectors.json'
# testing_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/test_feature_vectors.json'
# parameter_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/movie_review_small.NB'
# output_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/output.txt'

# global variable for num 
num = 2

"""
Function to initialize the classifier 
"""
def initialize_classifier():	
    print("Hello! Welcome to Naive Bayes Classifier for Movie Review Prediction")
    print("Please pick which a number for file you would like to run the program on")
    print("1. Small Movie Review Dataset with BOW Parameters")
    print("2. Movie Review Dataset with BOW Parameters")
    print("3. Movie Review Dataset with BOW features and Experimental Parameters")
    print("4. Your own input files")

    num = int(input())
    # small review 
    if num == 1:
        training_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/train_feature_vectors.json'
        testing_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/test_feature_vectors.json'
        parameter_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/movie_review_small.NB'
        output_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/output.txt'

    # movie review 
    if num == 2:
        training_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
        testing_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
        parameter_file = 'movie-review-HW2/movie-review-BOW.NB'
        output_file = 'movie-review-HW2/output.txt'   

    if num == 3:
        training_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
        testing_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
        parameter_file = 'movie-review-HW2/movie-review-BOW-experiment.NB'
        output_file = 'movie-review-HW2/experiment-output.txt'    
    
    if num == 4:
        print("Input path of training folder with label folders inside")
        training_file = str(input())
        while not path.exists(training_file):
            print("Not a valid path - please try again")
            training_file = str(input())
        
        print("Input path of testing folder with label folders inside")
        testing_file = str(input())
        while not path.exists(testing_file):
            print("Not a valid path - please try again")
            testing_file = str(input())
        
        print("Input path of parameter file - it should be a .NB format")
        parameter_file = str(input())
        while not path.exists(parameter_file):
            print("Not a valid path - please try again")
            parameter_file = str(input())

        print("Input path of output file - it should be a .txt format for easy reading")
        output_file = str(input())
        while not path.exists(output_file):
            print("Not a valid path - please try again")
            output_file = str(input())

    build_parameter_file(training_file, testing_file, parameter_file, output_file)




"""
Function used to build the model parameters that would store the model parameters in a file
"""
def build_parameter_file(training_file, testing_file, parameter_file, output_file):
    model_parameter_dict = dict()

    # open training file and read through it 
    f = open(training_file)
    training_data = json.load(f)
    f.close()

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
    
 
    # NOTE: CREATE MORE EXPERIMENTAL PARAMS HERE
    # TODO: CREATE EXPERIMENTAL PARAMS  


    # print(model_parameter_dict)
    # now to write to the parameter file 
    if not os.path.exists(parameter_file):
        try:
            os.makedirs(parameter_file)
        except Exception as e:
            print(e)
            raise

    param_filename = os.path.split(parameter_file)[1]
    param_path = os.path.dirname(os.path.normpath(parameter_file))
    with open(os.path.join(param_path, param_filename), 'w') as outfile:
        outfile.write(json.dumps(model_parameter_dict, indent=4))
    

    # call naive bayes 
    naive_bayes(parameter_file, testing_file, output_file, label_dict, num_vocab, num_of_words_in_class)



def write_to_output_file(output_file_dict, output_file, accuracy, num_correct, num_incorrect, num_of_test_docs):
    first_row_text = "   "
    count = 0

    with open(os.path.normpath(output_file), 'w') as outfile:
        for vector in output_file_dict:
            col_string = ""
            for column in output_file_dict[vector].items():

                if count == 0:
                    first_row_text += "|   " + str(column[0]) +  "    "
                col_string += "          " + str(column[1]) 
            
            if count == 0:
                outfile.write(first_row_text + "\n")
            outfile.write(col_string + "\n")
            count+=1
        
        accuracy_text = "Num correct : " + str(num_correct) + "\n" + "Num incorrect : " + str(num_incorrect) + "\n" + "Num of documents : " + str(num_of_test_docs) + "\n" + "Accuracy: " + str(accuracy) + " %\n"
        # outfile.write("Accuracy: " + str(accuracy) + " %\n")
        outfile.write("\n" + accuracy_text)
        
        # Also print it to the console 
        # output_file_end = os.
        print("----------------------------------------------")
        print("Evaluation of the Naive Bayes Model on Dataset")
        print("----------------------------------------------")
        print(f"The results have been written to {os.path.basename(os.path.normpath(output_file))}")
        print(accuracy_text)
            
                


def naive_bayes(model_parameter_dict, testing_file, output_prediction_file, label_dict, num_vocab, num_of_words_in_class):
    output_file_dict = dict()
    num_correct = 0
    num_incorrect = 0
    example_num = 1
    parameter_dict = dict()
    
    f = open(testing_file)
    testing_data = json.load(f)
    f.close()

    f = open(model_parameter_dict)
    parameter_dict = json.load(f)
    f.close()

    num_of_test_docs = len(testing_data)

    # go through each file 
    for vector in testing_data:
        max_val = -10000000
        predicted_label = ""
        vector_dict = dict()
        vector_dict["Example"] = example_num
        
        for key, value in vector.items():
            for label in label_dict:
                label_col_name = label + " Prediction"
                label_prob_name  = "P("+ label + ")"
                for word, word_count in value.items():
                    prob_name = "P(" +  word   + "|" + label + ")"

                    # this is to deal with test words not in train
                    if prob_name not in parameter_dict:
                        parameter_dict[prob_name] = 1 / (num_of_words_in_class[label] + num_vocab)


                    if label_col_name in vector_dict:
                        param_val = parameter_dict[prob_name]
                        vector_dict[label_col_name] += math.log10(param_val)
                    else:
                        vector_dict[label_col_name] = math.log10((parameter_dict[label_prob_name])) + math.log10((parameter_dict[prob_name]))
                

                if label_col_name in vector_dict and vector_dict[label_col_name] > max_val:
                    max_val = vector_dict[label_col_name]
                    predicted_label = label
            
            vector_dict["Predicted Label"] = predicted_label
            actual_label = key
            vector_dict["Actual Label"] = key

            if vector_dict["Predicted Label"] == vector_dict["Actual Label"]:
                num_correct += 1
            else:
                num_incorrect +=1

        output_file_dict[example_num] = vector_dict
        example_num += 1  

    # calculate the accuracy 
    accuracy = (num_correct / len(testing_data)) * 100
    # Now to write to output file 
    write_to_output_file(output_file_dict, output_prediction_file, accuracy, num_correct, num_incorrect, num_of_test_docs)


# build_parameter_file(training_file, testing_file, parameter_file, output_file)


initialize_classifier()