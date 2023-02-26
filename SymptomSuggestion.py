import nltk

# Predicts diseases based on the symptoms entered and selected by the user.
# importing all necessary libraries
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
from xgboost import XGBClassifier
import math
from Treatment import diseaseDetail
from sklearn.linear_model import LogisticRegression

warnings.simplefilter("ignore")

nltk.download('all')

global found_symptoms


def twilioInputSymptoms(twilprompt):
    global found_symptoms
    """# Symptoms initially taken from user."""
    # Taking symptoms from user as input
    user_symptoms = str(twilprompt)[1:-1].lower().split(',')
    # Preprocessing the input symptoms
    processed_user_symptoms = []
    for sym in user_symptoms:
        sym = sym.strip()
        sym = sym.replace('-', ' ')
        sym = sym.replace("'", '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)

    """Pre-processing on symptoms entered by user is done."""

    # Taking each user symptom and finding all its synonyms and appending it to the pre-processed symptom string
    user_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym) + 1):
            for subset in combinations(user_sym, comb):
                subset = ' '.join(subset)
                subset = synonyms(subset)
                str_sym.update(subset)
        str_sym.add(' '.join(user_sym))
        user_symptoms.append(' '.join(str_sym).replace('_', ' '))
    # query expansion performed by joining synonyms found for each symptom initially entered
    print("After query expansion done by using the symptoms entered")
    print(user_symptoms)

    """The below procedure is performed in order to show the symptom synonyms found for the symptoms entered by the user.

    The symptom synonyms and user symptoms are matched with the symptoms present in dataset. Only the symptoms which matches the symptoms present in dataset are shown back to the user.
    """

    # Loop over all the symptoms in dataset and check its similarity score to the synonym string of the user-input
    # symptoms. If similarity>0.5, add the symptom to the final list
    draft_found_symptoms = set()
    for idx, data_sym in enumerate(dataset_symptoms):
        data_sym_split = data_sym.split()
        for user_sym in user_symptoms:
            count = 0
            for symp in data_sym_split:
                if symp in user_sym.split():
                    count += 1
            if count / len(data_sym_split) > 0.5:
                draft_found_symptoms.add(data_sym)
    found_symptoms = list(draft_found_symptoms)

    """## **Prompt the user to select the relevant symptoms by entering the corresponding indices.**"""

    # Print all found symptoms
    response = "Top matching symptoms from your search!\n"
    for idx, symp in enumerate(found_symptoms):
        print(idx, ":", symp)
        response = response + str(idx) + ":" + symp + "\n"
    response += "   \nPlease select the relevant symptoms. Enter indices (separated-space):\n"
    return response


"""**synonyms function** finds the synonymous terms of a symptom entered by the user.

This is necessary as the user may use a term for a symptom which may be different from the one present in dataset.
This improves the accuracy by reducing the wrong predictions even when symptoms for a disease are entered slightly different than the ones on which model is trained.

*Synonyms are searched on Thesaurus.com and NLTK Wordnet*
"""


# returns the list of synonyms of the input word from thesaurus.com (https://www.thesaurus.com/) and wordnet (https://www.nltk.org/howto/wordnet.html)
def synonyms(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'})
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)


# utlities for pre-processing
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

"""**Disease Symptom dataset** was created in a separate python program.

**Dataset scrapping** was done using **NHP website** and **wikipedia data**

Disease Combination dataset contains the combinations for each of the disease present in dataset as practically it is often observed that it is not necessary for a person to have a disease when all the symptoms are faced by the patient or the user.

*To tackle this problem, combinations are made with the symptoms for each disease.*

 **This increases the size of the data exponentially and helps the model to predict the disease with much better accuracy.**

*df_comb -> Dataframe consisting of dataset generated by combining symptoms for each disease.*

*df_norm -> Dataframe consisting of dataset which contains a single row for each diseases with all the symptoms for that corresponding disease.*

**Dataset contains 261 diseases and their symptoms**
"""

# Load Dataset scraped from NHP (https://www.nhp.gov.in/disease-a-z) & Wikipedia
# Scrapping and creation of dataset csv is done in a separate program
df_comb = pd.read_csv("../Dataset/dis_sym_dataset_comb.csv")  # Disease combination
df_norm = pd.read_csv("../Dataset/dis_sym_dataset_norm.csv")  # Individual Disease

X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]

"""Using **Logistic Regression (LR) Classifier** as it gives better accuracy compared to other classification models as observed in the comparison of model accuracies in Model_latest.py

Cross validation is done on dataset with cv = 5
"""

lr = LogisticRegression()
lr = lr.fit(X, Y)
scores = cross_val_score(lr, X, Y, cv=5)

X = df_norm.iloc[:, 1:]
Y = df_norm.iloc[:, 0:1]

# List of symptoms
dataset_symptoms = list(X.columns)

final_symp = []


def twilioProcessSymptoms(mostRelevant):
    global found_symptoms
    # Show the related symptoms found in the dataset and ask user to select among them
    select_list = mostRelevant.split()

    # Find other relevant symptoms from the dataset based on user symptoms based on the highest co-occurance with the
    # ones that is input by the user
    dis_list = set()
    counter_list = []
    for idx in select_list:
        symp = found_symptoms[int(idx)]
        final_symp.append(symp)
        dis_list.update(set(df_norm[df_norm[symp] == 1]['label_dis']))

    for dis in dis_list:
        row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
        row[0].pop(0)
        for idx, val in enumerate(row[0]):
            if val != 0 and dataset_symptoms[idx] not in final_symp:
                counter_list.append(dataset_symptoms[idx])


global diseases
global topk_index_mapping


def twilioPrintSymptoms():
    global diseases
    global topk_index_mapping
    """Final Symptom list"""
    # Create query vector based on symptoms selected by the user
    response = ""
    # response = "\nFinal list of Symptoms that will be used for prediction:"
    sample_x = [0 for x in range(0, len(dataset_symptoms))]
    for val in final_symp:
        response += val
        sample_x[dataset_symptoms.index(val)] = 1

    """Prediction of disease is done"""

    # Predict disease
    lr = LogisticRegression()
    lr = lr.fit(X, Y)
    prediction = lr.predict_proba([sample_x])

    """Show top k diseases and their probabilities to the user.
    
    K in this case is 10
    """

    k = 10
    diseases = list(set(Y['label_dis']))
    diseases.sort()
    topk = prediction[0].argsort()[-k:][::-1]

    """# **Showing the list of top k diseases to the user with their prediction probabilities.**
    
    # **For getting information about the suggested treatments, user can enter the corresponding index to know more details.**
    """

    response += f"\nTop {k} diseases predicted based on symptoms"
    topk_dict = {}
    # Show top 10 highly probable disease to the user.
    for idx, t in enumerate(topk):
        match_sym = set()
        row = df_norm.loc[df_norm['label_dis'] == diseases[t]].values.tolist()
        row[0].pop(0)

        for idx, val in enumerate(row[0]):
            if val != 0:
                match_sym.add(dataset_symptoms[idx])
        prob = (len(match_sym.intersection(set(final_symp))) + 1) / (len(set(final_symp)) + 1)
        prob *= mean(scores)
        topk_dict[t] = prob
    j = 0
    topk_index_mapping = {}
    topk_sorted = dict(sorted(topk_dict.items(), key=lambda kv: kv[1], reverse=True))
    for key in topk_sorted:
        prob = topk_sorted[key] * 100
        response += str(j) + " Disease name:" + str(diseases[key]) + "\tProbability:" + str(round(prob, 2)) + "%"
        topk_index_mapping[j] = key
        j += 1
    return response
