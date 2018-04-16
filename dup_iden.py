import pandas as pd
import numpy as np
from nltk.corpus import stopwords # Natural Language Toolkit or NLTK is a suite of libraries and programs for symbolic and statistical natural language processing& it provides libraries for classification, tokenization and so on.

stop_words = set(stopwords.words("english")) # Extract all the stop words in english langauge into a list.

#training

train_data = pd.read_csv('train.csv') # Read training dataset train.csv

def compute_score(row): # take a pair of questions and compute Jaccard similarity score between the questions.
    Ques1_list = []
    Ques2_list = []   

    # Append all the words except stop words in question1 into a list Ques1_list
    for word in str(row['question1']).lower().split():
        if word not in stop_words:
            Ques1_list.append(word)
    # Append all the words except stop words in question2 into a list Ques2_list          
    for word in str(row['question2']).lower().split():
        if word not in stop_words:
            Ques2_list.append(word)

    # set operations are performed to calculate jaccard score
    no_similar_words = len(set(Ques1_list) & set(Ques2_list)) #The intersection operation is carried out to get the words that similar in the both questions and it's count is taken.
    no_total_words= len(set(Ques1_list) | set(Ques2_list)) #The union operation is performed to get the total words in the two questions and it's count is taken.
    if len(Ques1_list) == 0 or len(Ques2_list) == 0:
        return 0
    #calculate the Jaccard similarity score corresponds to the pair of questions
    jac_score = float(no_similar_words)/float(no_total_words) 
    return jac_score

# compute the jaccard for every pair of questions and store the value into score_list
score_list = []
for index, row in train_data.iterrows():
    jac_score = compute_score(row)
    score_list.append(jac_score)

train_data['jaccard_score'] = score_list
A = np.array([score_list, np.ones(len(score_list))])
w = np.linalg.lstsq(A.T,train_data['is_duplicate'])[0]

#training ends

#testing

test_data = pd.read_csv('test.csv') # Read test dataset test.csv
jac_score_list = [] 

#calculate the jaccard similarity score for every pair of questions in the test dataset
for index, row in test_data.iterrows(): 
    jac_score = compute_score(row)
    jac_score_list.append(jac_score)


test_data['jaccard_score'] = jac_score_list
test_data['duplicate_index'] = test_data['jaccard_score']*w[0]+w[1]


sub = pd.DataFrame() #write the output into a file output.csv
sub['test_id'] = test_data['test_id']
sub['duplicate_index'] = test_data['duplicate_index']
sub.to_csv('output.csv', index=False)

#testing ends


