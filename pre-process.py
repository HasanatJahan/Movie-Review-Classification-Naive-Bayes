"""
Name: Hasanat Jahan
Homework 2 Naive Bayes Implementation on Movie Review 
"""


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
# for train
# file_path = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/data/train'

# for test
file_path = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/data/test'


main_directory_path = '/Users/jahan/Desktop/CS381/Homework2/small_movie_review/feature_vectors'

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
    
    # remove punctuation
    table_ = str.maketrans('', '', punctuation_list)
    modified_input_text = input_text.translate(table_)

    # create input word list 
    input_list = modified_input_text.split(" ")

    for word in input_list:
        # this part for the larger processing 
        # NOTE: don't need this if word in vocab_dict:
        #     word = word.strip('\n')
        #     lower_word = word.lower()
        #     output_text_list.append(lower_word)

        # this part is only for the small review preprocessing
        if word != "\n":
            word = word.strip('\n')
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
    
    # labels = ["pos", "neg"]
    # labels = ["comedy", "action"]
    labels = find_labels(file_path)

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
    # the file name would be different for movie review 
    json_filename = type_of_data + '_feature_vectors.json'

    write_json(main_directory_path, json_filename, feature_vectors)


"""
Now to call all necessary functions for preprocessing 
"""
preprocess(vocab_dict)



