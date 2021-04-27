"""
Name: Hasanat Jahan
Homework 2 Naive Bayes Implementation on Movie Review 
"""

import os
import json
import string 

# note: change this to take input from terminal regarding file name 
vocab_file_path = 'movie-review-HW2/aclImdb/imdb.vocab'
punctuation_list = string.punctuation 
# for training data 
file_path = 'movie-review-HW2/aclImdb/train'

# for test data 
# file_path = 'movie-review-HW2/aclImdb/test'
main_directory_path = '/Users/jahan/Desktop/CS381/Homework2/movie-review-HW2/feature_vectors'

# from here you have the test small files 
# for train
# file_path = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/data/train'
# for test
# file_path = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/data/test'
# main_directory_path = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors'

"""
Open the vocabulary file 
"""
def get_vocab_dict(vocab_file_path):
    vocab_dict = dict()
    with open(vocab_file_path) as f:
        vocab_list = f.read().split()
    for word in vocab_list:
        vocab_dict[word] = ""
    return vocab_dict

vocab_dict = get_vocab_dict(vocab_file_path)
# print(vocab_dict)

"""
Create a dictionary of word counts based on word list created from input text
"""
def create_word_count_dict(input_text, vocab_dict):

    # remove punctuation
    table_ = str.maketrans('', '', punctuation_list)
    modified_input_text = input_text.translate(table_)

    # create input word list 
    input_list = modified_input_text.split(" ")
    input_word_dict = dict()

    redundant_words_dict = {
        "the" : "", 
        "a": "", 
        "an": "", 
        "from": "", 
        "across": "", 
        "along": "", 
        "in": "", 
        "by": "", 
        "upon": "", 
        "with": "", 
        "within": "", 
        "of": "", 
        "to": "" 
    }

    for word in input_list:
        # this is for normal BOW for all other options
        word = word.strip('\n')

        # if word in vocab_dict:
        # this is for removing redundant words
        if word in vocab_dict and word not in redundant_words_dict:
            lower_word = word.lower()
            if word not in input_word_dict:
                input_word_dict[word] = 1
            else:
                input_word_dict[word] += 1  

    return input_word_dict


"""
Reference: https://stackoverflow.com/questions/64196315/json-dump-into-specific-folder
"""
def write_json(target_path, target_file, feature_vectors):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise 
    with open(os.path.join(target_path, target_file), 'w') as outfile:
        outfile.write(json.dumps(feature_vectors, indent=4))

"""
Find the directory contents- find the directory name for labels
"""
def find_labels(file_path):
    labels = []
    directory_contents = os.listdir(file_path)
    # print(directory_contents)
    for item in directory_contents:
        if os.path.isdir(os.path.join(file_path, item)):
            labels.append(item)
    return labels

"""
Now for the preprocess function 
"""
def preprocess(vocab_dict):
    feature_vectors = []
    document_text = []

    # labels = ["pos", "neg"]
    labels = find_labels(file_path)

    type_of_data = os.path.basename(os.path.normpath(file_path))
    for dirname, _, filenames in os.walk(file_path):
        label = os.path.basename(os.path.normpath(dirname))
        for filename in filenames:
            if filename.endswith('.txt') and label in labels:
                file_name = os.path.join(dirname, filename)
                with open(file_name, "r") as file:
                    document_text.append((label, file.read()))

    for document in document_text:
        text_of_file = document[1]
        feature_dict = create_word_count_dict(text_of_file, vocab_dict)
        label = document[0]
        feature_vectors.append({label: feature_dict})
    
    
    # after all the files have been read 
    # the file name would be different for movie review 


    # json_filename = type_of_data + '_feature_vectors.json'

    # filename for experimental features
    json_filename = "experimental_" + type_of_data + '_feature_vectors.json'

    write_json(main_directory_path, json_filename, feature_vectors)


"""
Now to call all necessary functions for preprocessing 
"""
preprocess(vocab_dict)



