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

# trying with the small file 
training_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/train_feature_vectors.json'
testing_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors/test_feature_vectors.json'
parameter_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/movie_review_small.NB'
output_file = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/output.txt'


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
    naive_bayes(parameter_file, testing_file, output_file, label_dict)

    


def naive_bayes(model_parameter_dict, testing_file, output_prediction_file, label_dict):
    # NOTE: THE TESTING FILE CONTAINS THE FEATURE VECTORS
    # NOTE: PLAN - DO NAIVE BAYES FOR BOTH LABELS - CREATE A GENERAL NAIVE BAYES FUNCTION
    # AND THEN COMPARE THE VALUE RETURNED BY BOTH LABELS AND PICK THE ONE WITH THE LARGER VALUE 
    # THEN OUTPUT THE PREDICTED LABEL AND OUTPUT LABEL FOR ALL 
    # NOTE: ALSO INLCUDE THE PROBABILITIES OF EACH CLASS 


    # NOTE: FIND A WAY TO DO ACCURACY 
    # label_evaluation = z
    # label evaluation has document number,  
    # prob of each label, 
    # predicted label 
    # actual label,

    output_file_dict = dict()
    num_correct = 0
    num_of_test_docs = len(testing_file)
    example_num = 1
    parameter_dict = dict()
    
    f = open(testing_file)
    testing_data = json.load(f)
    f.close()

    f = open(model_parameter_dict)
    parameter_dict = json.load(f)
    f.close



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
                    if label_col_name in vector_dict:

                        # NOTE: checking the dict is only added in for the small movie review 
                        # words not in vocab should be added in with 0 
                        # ONLY FOR THIS ONE SMALL - DELETE BLOCK LATER 
                        if prob_name not in parameter_dict and label == "comedy":
                            param_val = 1/16
                        elif prob_name not in parameter_dict and label == "action":
                            param_val = 1/18
                        else:
                            param_val = parameter_dict[prob_name]
                        
                        vector_dict[label_col_name] += math.log10(param_val)

                        # vector_dict[label_col_name] += math.log10(parameter_dict[prob_name])


                    else:
                        vector_dict[label_col_name] = math.log10((parameter_dict[label_prob_name])) + math.log10((parameter_dict[prob_name]))

                if vector_dict[label_col_name] > max_val:
                    max_val = vector_dict[label_col_name]
                    predicted_label = label
            
            vector_dict["Predicted Label"] = predicted_label
            actual_label = key
            vector_dict["Actual Label"] = key

        output_file_dict[example_num] = vector_dict
        example_num += 1  

    print(output_file_dict)




build_parameter_file(training_file, testing_file, parameter_file, output_file)

