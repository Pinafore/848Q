from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
import os

from typing import Union, Dict
from collections.abc import Iterable

import math
import logging
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
QN_PATH = 'questions.pickle'
ANS_PATH = 'answers.pickle'

import os

from nltk.tokenize import sent_tokenize
from guesser import print_guess, Guesser

kTFIDF_TEST_QUESTIONS = {"This capital of England": ['Maine', 'Boston'],
                        "The author of Pride and Prejudice": ['Jane_Austen', 'Jane_Austen'],
                        "The composer of the Magic Flute": ['Wolfgang_Amadeus_Mozart', 'Wolfgang_Amadeus_Mozart'],
                        "The economic law that says 'good money drives out bad'": ["Gresham's_law", "Gresham's_law"],
                        "located outside Boston, the oldest University in the United States": ['College_of_William_&_Mary', 'Rhode_Island']}


class DummyVectorizer:
    """
    A dumb vectorizer that only creates a random matrix instead of something real.
    """
    def __init__(self, width:int=50):
        self.width = width
        self.vocabulary_ = {}
    
    def transform(self, questions: Iterable):
        import numpy as np
        return np.random.rand(len(questions), self.width)

class TfidfGuesser(Guesser):
    """
    Class that, given a query, finds the most similar question to it.
    """
    def __init__(self, filename:str, min_df:int=10, max_df:float=0.4):
        """
        Initializes data structures that will be useful later.

        Args:
           filename: base of filename we store vectorizer and documents to
           min_df: we use the sklearn vectorizer parameters, this for min doc freq
           max_df: we use the sklearn vectorizer parameters, this for max doc freq
        """

        # You'll need add the vectorizer here and replace this fake vectorizer
        self.tfidf_vectorizer = DummyVectorizer()
        self.tfidf = None 
        self.questions = None
        self.answers = None
        self.filename = filename

    def train(self, training_data, answer_field='page', split_by_sentence=True,
                    min_length=-1, max_length=-1, remove_missing_pages=True):
        """
        The base class (Guesser) populates the questions member, so
        all that's left for this function to do is to create new members
        that have a vectorizer (mapping documents to tf-idf vectors) and
        the matrix representation of the documents (tfidf) consistent
        with that vectorizer.
        """
        
        Guesser.train(self, training_data, answer_field, split_by_sentence, min_length,
                      max_length, remove_missing_pages)

        self.tfidf = self.tfidf_vectorizer.transform(self.questions)
        logging.info("Creating tf-idf dataframe with %i" % len(self.questions))
        
    def save(self):
        """
        Save the parameters to disk
        """
        Guesser.save_questions_and_answers(self)
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open("%s.tfidf.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf, f)

    def __call__(self, question, max_n_guesses):
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        question -- Raw text of the question
        max_n_guesses -- How many top guesses to return
        """
        top_questions = []
        top_answers = []
        top_sim = []

        # Compute the cosine similarity
        question_tfidf = self.tfidf_vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, self.tfidf)
        cos = cosine_similarities[0]
        indices = cos.argsort()[::-1]
        guesses = []
        for i in range(max_n_guesses):
            # The line below is wrong but lets the code run for the homework.
            # Remove it or fix it!
            idx = i
            guess =  {"question": self.questions[idx], "guess": self.answers[idx],
                      "confidence": cos[idx]}
            guesses.append(guess)
        assert len(guesses) <= max_n_guesses, "Too many guesses: %i > %i" % (len(guesses), max_n_guesses)
        return guesses
    
    def load(self):
        """
        Load the tf-idf guesser from a file
        """
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open("%s.tfidf.pkl" % path, 'rb') as f:
            self.tfidf = pickle.load(f)

        self.load_questions_and_answers()

if __name__ == "__main__":
    # Load a tf-idf guesser and run it on some questions
    from parameters import *
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    add_general_params(parser)
    add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    
    guesser = load_guesser(flags, load=True)

    questions = list(kTFIDF_TEST_QUESTIONS.keys())
    guesses = guesser.batch_guess(questions, 3, 2)

    for qq, gg in zip(questions, guesses):
        print("----------------------")
        print(qq, gg)
