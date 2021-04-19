"""
    Pre-processing: prior to building feature vectors, 
    1. you should separate punctuation from words 
    2. lowercase the words in the reviews. 
    3. You will train NB classifier on the training partition using the BOW features 
    4. (use add-one smoothing, as we did in class). 
    You will evaluate your classifier on the test partition. In addition to BOW features, you should experiment with addi- tional features. In that case, please provide a description of the features in your report. Save the parameters of your BOW model in a file called movie- review-BOW.NB. Report the accuracy of your program on the test data with BOW features.

"""
import os

"""
Open the vocabulary file 
"""
def get_vocab_dict(vocab_file_path):
    vocab_dict = dict()
    with open(vocab_file_path) as f:
        vocab_dict = f.read().split()
    return vocab_dict

vocab_file_path = 'movie-review-HW2/aclImdb/imdb.vocab'
vocab_dict = get_vocab_dict(vocab_file_path)
print(vocab_dict)

"""
Create a dictionary of 
"""
def create_word_count_dict(input_text):
    input_word_dict = dict()
    for word in input_text:
        if word not in input_word_dict:
            input_word_dict[word] = 1
        else:
            input_word_dict[word] += 1   
    return input_word_dict


"""
Takes input text and creates word list 
"""
def create_word_list(input_text, vocab):
    output_text_list = []
    # identify punctuation to remove 
    punctuation_to_remove = {'"', '"', '!', ':', ',', '.', ';', '(', ')', '{', '}', '[', ']', '/','\'', '\\'}
    
    for word in input_text:
        if word not in punctuation_to_remove:
            lower_word = word.lower()
            output_text_list.append(lower_word)

    return output_text_list


"""
"""




