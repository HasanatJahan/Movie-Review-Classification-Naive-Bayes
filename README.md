# Movie Review Classification with Naive Bayes Bag-of-Word Features 

## Pre-requisites 
1. Must have Python 3 installed 
2. Relevant folders and files should have the structure:

### NB.py 
- training folder:
    - train folder should have folders for each class and within each folder there should be documents for each class 
    for example: for positive and negative classes, the "train" folder should have folders within it named "pos" and "neg"
- testing folders:
    - similarly test folder should have folders for each class and within each folder there should be documents for each class 
    for example: for positive and negative classes, the "test" folder should have folders within it named "pos" and "neg"
- parameter file:
    - parameter file should have the extension .NB and the path should be specified if asked in input 
- output file 
    - output file should have the extension .txt and the path should be specified if asked in input 
 
### Preprocess.py 
This file has been used to create feature vectors based on the specific files given for the project. However, it can be generally used. If the variable 'main_directory_path' is 
modified to include the path of the file where you would like to save your feature vectors and the variable 'file_path' is used to specify the folder of documents you'd like to run 
preprocessing on, and variable 'vocab_file_path' is changed to your desired .vocab file path, then it can generally be used to preprocess document folders into feature vectors. 


## To run for output and view
1. Open Terminal and navigate to directory containing  `NB.py`, this directory should also include the folder 'movie-review-HW2' and 'small_movie_review' as well as `preprocess.py`
2. In terminal, type in below command to run `NB.py` in Python3
```
python3 NB.py
```
3. The terminal then presents the following options for Naive Bayes classification:
```
Hello! Welcome to Naive Bayes Classifier for Movie Review Prediction
Please pick which a number for the investigation results you want to explore
1. Small Movie Genre Determination with BOW Parameters
2. Movie Review Classification with BOW Parameters
3. Movie Review Classification with added Experimental Binary Word Count
4. Movie Review Classification with added Experimental Removal of Redundant Words
5. Movie Review Classification with BOW Parameters and document word count parameter
6. Your own input files
```
Then enter your desired number and press enter for results 
The accuracy results will present in the terminal and the individual results for files are present in the text file specified by the terminal. 
An example output:
```
----------------------------------------------
Evaluation of the Naive Bayes Model on Dataset
----------------------------------------------
The results have been written to output.txt
Num correct : 22273
Num incorrect : 2727
Num of documents : 25000
Accuracy: 89.092 %
```
For option 6, the terminal will walk the user through which file they want for each parameter of the Naive Bayes classifier and then run accordingly. The files input for feature vectors must first be pre-processed by `pre-process.py`

#### To view output file 
To see the relevant output txt file, navigate to mentioned file in the results either from terminal or through GUI and open to see results. 
Example of top 4 results: 
```
   |   Example    |   neg Prediction    |   pos Prediction    |   Predicted Label    |   Actual Label    
          1          -104.44693977325286          -103.55597007605664          pos          neg
          2          -384.9458345541981          -393.5187105111109          neg          neg
          3          -249.25340568899088          -254.1831343339639          neg          neg

```

Example of bottom 5 results:
```
          25000          -262.8727955926842          -258.14903870174277          pos          pos

Num correct : 22089
Num incorrect : 2911
Num of documents : 25000
Accuracy: 88.356 %

```


