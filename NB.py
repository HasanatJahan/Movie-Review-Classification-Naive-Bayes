"""
Name: Hasanat Jahan
Homework 2 Naive Bayes Implementation on Movie Review 
"""

import json
import os 
import math


"""
Function to initialize the classifier 
"""
def initialize_classifier():	
    print("Hello! Welcome to Naive Bayes Classifier for Movie Review Prediction")
    print("Please pick which a number for the investigation results you want to explore")
    print("1. Small Movie Genre Determination with BOW Parameters")
    print("2. Movie Review Classification with BOW Parameters")
    print("3. Movie Review Classification with added Experimental Binary Word Count")
    print("4. Movie Review Classification with added Experimental Removal of Redundant Words")
    print("5. Movie Review Classification with BOW Parameters and document word count parameter")
    print("6. Your own input files")


    num = int(input())
    while type(num) != int or num < 1 or num > 6:
        print("Please enter a valid number between 1 to 6")
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

    # BINARY COUNTS
    if num == 3:
        training_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
        testing_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
        parameter_file = 'movie-review-HW2/movie-review-BOW.NB'
        output_file = 'movie-review-HW2/experiment-output.txt'    
    
    # REDUNDANT WORD REMOVAL
    if num == 4:
        training_file = 'movie-review-HW2/feature_vectors/experimental_train_feature_vectors.json'
        testing_file = 'movie-review-HW2/feature_vectors/experimental_test_feature_vectors.json'
        parameter_file = 'movie-review-HW2/movie-review-BOW.NB'
        output_file = 'movie-review-HW2/experiment-output-1.txt'

    # ADDING THE EXTRA FEATURE OF WORD COUNT PER DOCUMENT 
    if num == 5:
        training_file = 'movie-review-HW2/feature_vectors/train_feature_vectors.json'
        testing_file = 'movie-review-HW2/feature_vectors/test_feature_vectors.json'
        parameter_file = 'movie-review-HW2/movie-review-BOW.NB'
        output_file = 'movie-review-HW2/experiment-output-2.txt'


    # USER FILE INPUT
    if num == 6:
        print("Input path of training folder with label folders inside")
        training_file = str(input())
        while not os.path.exists(training_file):
            print("Not a valid path - please try again")
            training_file = str(input())
        
        print("Input path of testing folder with label folders inside")
        testing_file = str(input())
        while not os.path.exists(testing_file):
            print("Not a valid path - please try again")
            testing_file = str(input())
        
        print("Input path of parameter file - it should be a .NB format")
        parameter_file = str(input())
        while not os.path.exists(parameter_file):
            print("Not a valid path - please try again")
            parameter_file = str(input())

        print("Input path of output file - it should be a .txt format for easy reading")
        output_file = str(input())
        while not os.path.exists(output_file):
            print("Not a valid path - please try again")
            output_file = str(input())

    train_naive_bayes(training_file, testing_file, parameter_file, output_file, num)




"""
Function used to build the model parameters that would store the model parameters in a file
"""
def train_naive_bayes(training_file, testing_file, parameter_file, output_file, user_input_option):
    model_parameter_dict = dict()

    # open training file and read through it 
    f = open(training_file)
    training_data = json.load(f)
    f.close()

    # get the number of documents 
    num_document = len(training_data)
    label_dict = dict()
    # holds the general word count in each class 
    class_BOW = dict()
    num_of_words_in_class = dict()
    vocab_dict = dict()

    # iterate through the dictionary to draw out the labels to create prior probability 
    # for each vector 
    for i in training_data:
        # inside the vector - it's a dictionary
        for key, value in i.items():
            if key not in label_dict:
                label_dict[key] = 1 
                # declare an empty dict here 
                class_BOW[key] = {}               
            else:
                label_dict[key] += 1 
            
            # for each word in the vector 
            for word, word_count in value.items():
                # for all evaluations other than binary 
                if user_input_option != 3: 
                    # populate teh number of words in the class 
                    if key in num_of_words_in_class:
                        num_of_words_in_class[key] += word_count
                    else:
                        num_of_words_in_class[key] = word_count
                    
                    # create a total word dictionary 
                    if word not in vocab_dict:
                        vocab_dict[word] = word_count
                    else:
                        vocab_dict[word] += word_count

                    # create number of words per class 
                    if word not in class_BOW[key]:
                        class_BOW[key][word] = word_count
                    else:
                        class_BOW[key][word] += word_count
                
                # for binary naive bayes 
                else:
                    # populate teh number of words in the class 
                    if key in num_of_words_in_class:
                        num_of_words_in_class[key] += 1
                    else:
                        num_of_words_in_class[key] = 1
                    
                    # create a total word dictionary 
                    if word not in vocab_dict:
                        vocab_dict[word] = 1
                    else:
                        vocab_dict[word] += 1

                    # number of words per class     

                    if word not in class_BOW[key]:
                        class_BOW[key][word] = 1
                    else:
                        class_BOW[key][word] += 1

    # get the number of vocabulary words in training
    num_vocab = len(vocab_dict)

    # print(f"This is num vocab {num_vocab}")
    # print(f"classBOW {class_BOW}")
    # print(f"Number of words in class {num_of_words_in_class}")
    # print(f"Label dict {label_dict}")
    # print(f"vocab dict {vocab_dict}")
    # print(f"num of vocab words {len(vocab_dict)}")

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
    param_path = os.path.dirname(os.path.normpath(parameter_file))
    with open(os.path.join(param_path, param_filename), 'w') as outfile:
        outfile.write(json.dumps(model_parameter_dict, indent=4))
    

    # call naive bayes 
    test_naive_bayes(parameter_file, testing_file, output_file, label_dict, num_vocab, num_of_words_in_class, user_input_option)



def write_to_output_file(output_file_dict, output_file, accuracy, num_correct, num_incorrect, num_of_test_docs):
    first_row_text = "   "
    count = 0

    with open(os.path.normpath(output_file), 'w') as outfile:
        for vector in output_file_dict:
            col_string = ""
            for column in output_file_dict[vector].items():
                # top row in output        
                if count == 0:
                    first_row_text += "|   " + str(column[0]) +  "    "
                # for all values in all files 
                col_string += "          " + str(column[1]) 
            
            if count == 0:
                outfile.write(first_row_text + "\n")
            outfile.write(col_string + "\n")
            count+=1
        
        accuracy_text = "Num correct : " + str(num_correct) + "\n" + "Num incorrect : " + str(num_incorrect) + "\n" + "Num of documents : " + str(num_of_test_docs) + "\n" + "Accuracy: " + str(accuracy) + " %\n"
        # outfile.write("Accuracy: " + str(accuracy) + " %\n")
        outfile.write("\n" + accuracy_text)
        
        # Also print it to the console 
        print("----------------------------------------------")
        print("Evaluation of the Naive Bayes Model on Dataset")
        print("----------------------------------------------")
        print(f"The results have been written to {os.path.basename(os.path.normpath(output_file))}")
        print(accuracy_text)
            
                


def test_naive_bayes(model_parameter_file, testing_file, output_prediction_file, label_dict, num_vocab, num_of_words_in_class, user_input_option):
    output_file_dict = dict()
    num_correct = 0
    num_incorrect = 0
    example_num = 1
    parameter_dict = dict()

    # hold the incorrectly predicted vectors for later analysis 
    incorrect_prediction_vectors = []
    correct_prediction_vectors = []

    f = open(testing_file)
    testing_data = json.load(f)
    f.close()

    f = open(model_parameter_file)
    parameter_dict = json.load(f)
    f.close()

    num_of_test_docs = len(testing_data)

    # go through each file 
    for vector in testing_data:
        max_val = -10000000
        predicted_label = ""

        # this is to hold the information about this vector 
        vector_dict = dict()
        vector_dict["Example"] = example_num

        # word_account = dict()

        for key, value in vector.items():

            # number of words per document
            num_of_words_in_doc = sum(value.values())

            # make a prediction for each label for a particular vector 
            for label in label_dict:
                label_col_name = label + " Prediction"
                label_prob_name  = "P("+ label + ")"

                for word, word_count in value.items():
                    prob_name = "P(" +  word   + "|" + label + ")"

                    # this is to deal with test words not in training
                    if prob_name not in parameter_dict:
                        parameter_dict[prob_name] = 1 / (num_of_words_in_class[label] + num_vocab)

                    if label_col_name in vector_dict:
                        param_val = parameter_dict[prob_name]
                        vector_dict[label_col_name] += math.log10(param_val)
                    else:
                        # this is the first prob of all naive bayes classifier expect 
                        if user_input_option != 5:
                            vector_dict[label_col_name] = math.log10((parameter_dict[label_prob_name])) + (math.log10((parameter_dict[prob_name])) * word_count)
                        
                        # log of word count per document feature added 
                        else:
                            vector_dict[label_col_name] = math.log10((parameter_dict[label_prob_name])) + (math.log10((parameter_dict[prob_name])) * word_count) + math.log10(num_of_words_in_doc)


                if label_col_name in vector_dict and vector_dict[label_col_name] > max_val:
                    max_val = vector_dict[label_col_name]
                    predicted_label = label
            
            vector_dict["Predicted Label"] = predicted_label
            actual_label = key
            vector_dict["Actual Label"] = key

            if vector_dict["Predicted Label"] == vector_dict["Actual Label"]:
                num_correct += 1
                correct_prediction_vectors.append(vector)
            else:
                num_incorrect +=1
                incorrect_prediction_vectors.append(vector)

        output_file_dict[example_num] = vector_dict
        example_num += 1  

    # DECIDED TO GO WITH BOW FEATURES BY THEMSELVES 
    # if user_input_option == 2:
    #     analyse_incorrect_predictions(incorrect_prediction_vectors, correct_prediction_vectors)

    # if it's the small dataset then write to the file 
    if user_input_option == 1:
        col_string = ""
        first_row_text = ""
        count = 0
        with open(os.path.normpath(output_prediction_file), 'w') as outfile:
            for vector in output_file_dict:
                for column in output_file_dict[vector].items():

                    if count == 0:
                        first_row_text += "|   " + str(column[0]) +  "    "
                    col_string += "          " + str(column[1]) 
                
                if count == 0:
                    outfile.write(first_row_text + "\n")
                outfile.write(col_string + "\n")
                count+=1
        print("----------------------------------------------")
        print("Evaluation of the Naive Bayes Model on Dataset")
        print("----------------------------------------------")
        print(f"The results have been written to {os.path.normpath(output_prediction_file)}")
        print(first_row_text)
        print(col_string)

    # calculate the accuracy and write to file 
    elif user_input_option != 1:
        accuracy = (num_correct / len(testing_data)) * 100
        # Now to write to output file 
        write_to_output_file(output_file_dict, output_prediction_file, accuracy, num_correct, num_incorrect, num_of_test_docs)
    

def analyse_incorrect_predictions(incorrect_prediction_vectors, correct_prediction_vectors):
    vector_word_dict = dict()
    vector_word_dict_correct = dict()
    label_dict = dict()
    label_word_count = dict()
    num_examples_wrong = len(incorrect_prediction_vectors)
    num_examples_correct = len(correct_prediction_vectors)

    for example in incorrect_prediction_vectors:
        for vector in example.items():
            if vector[0] in label_dict:
                label_dict[vector[0]] += 1
            else:
                label_dict[vector[0]] = 1

            for word, word_count in vector[1].items():
                if word in vector_word_dict:
                    vector_word_dict[word] += word_count
                else:
                    vector_word_dict[word] = word_count


                label_word_count[vector[0]] = word_count

    for example in correct_prediction_vectors:
        for vector in example.items(): 
            for word, word_count in vector[1].items():
                if word in vector_word_dict_correct:
                    vector_word_dict_correct[word] += word_count
                else:
                    vector_word_dict_correct[word] = word_count  

    total_predicted_incorrect = sum(label_dict.values())
    positive_instances = 0
    negative_instances = 0
    for label, value in label_dict.items():
        calculate_percent = value/total_predicted_incorrect * 100 
        print(f"Percent of {label} incorrect {calculate_percent} %")
    
    average_word_count_per_example = sum(vector_word_dict.values()) / num_examples_wrong
    print(f"Average word count per example in incorrect classification {average_word_count_per_example}")

    average_word_count_per_example = sum(vector_word_dict_correct.values()) / num_examples_correct
    print(f"Average word count per example in correct classification {average_word_count_per_example}")


                

initialize_classifier()
