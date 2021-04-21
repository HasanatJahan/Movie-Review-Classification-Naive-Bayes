"""
    Pre-processing: prior to building feature vectors, 
    1. you should separate punctuation from words 
    2. lowercase the words in the reviews. 
    3. You will train NB classifier on the training partition using the BOW features 
    4. (use add-one smoothing, as we did in class). 
    You will evaluate your classifier on the test partition. In addition to BOW features, you should experiment with addi- tional features. In that case, please provide a description of the features in your report. Save the parameters of your BOW model in a file called movie- review-BOW.NB. Report the accuracy of your program on the test data with BOW features.

"""


import os
import json
import string 

# note: change this to take input from terminal regarding file name 
vocab_file_path = 'movie-review-HW2/aclImdb/imdb.vocab'
punctuation_list = string.punctuation 

# file_path = 'movie-review-HW2/aclImdb/train'
# labels = ["pos", "neg"]

# from here you have the test small files 
file_path = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/train'


"""
Open the vocabulary file 
"""
def get_vocab_dict(vocab_file_path):
    vocab_dict = dict()
    with open(vocab_file_path) as f:
        vocab_dict = f.read().split()
    return vocab_dict

vocab_dict = get_vocab_dict(vocab_file_path)

"""
Takes input text and creates word list 
"""
def create_word_list(input_text, vocab_dict):
    output_text_list = []
    
    table_ = str.maketrans('', '', punctuation_list)
    modified_input_text = input_text.translate(table_)

    input_list = modified_input_text.split(" ")

    for word in input_list:
        if word in vocab_dict:
            lower_word = word.lower()
            output_text_list.append(lower_word)

    return output_text_list


"""
Create a dictionary of word counts based on word list created from input text
"""
def create_word_count_dict(input_list):
    input_word_dict = dict()
    for word in input_list:
        if word not in input_word_dict:
            input_word_dict[word] = 1
        else:
            input_word_dict[word] += 1  

    print(input_word_dict) 
    return input_word_dict


"""
Now for the preprocess function 
"""
def preprocess(vocab_dict):
    feature_vectors = []
    # labels = ["pos", "neg"]
    labels = ["comedy", "action"]
    type_of_data = os.path.basename(os.path.normpath(file_path))
    for dirname, _, filenames in os.walk(file_path):
        label = os.path.basename(os.path.normpath(dirname))
        for filename in filenames:
            if filename.endswith('.txt') and label in labels:
                f = open(os.path.join(dirname, filename), "r")
                output_word_list = create_word_list(f.read(), vocab_dict)
                feature_dict = create_word_count_dict(output_word_list)
                feature_vectors.append({label: feature_dict})
                f.close()

    # after all the files have been read 
    json_filename = type_of_data + '_feature_vectors.txt'
    # now to write the feature vectors to the input file 
    with open(json_filename, 'w') as outfile:
        json.dump(feature_vectors, outfile)


preprocess(vocab_dict)



