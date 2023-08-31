# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from sklearn.metrics.pairwise import linear_kernel
import gzip
import json

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess):
        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess):
        # How many characters long is the question?
        yield ("char", log(1 + len(run)))

        # How many words long is the question?
        yield ("word", log(1 + len(run.split())))

        # How many characters long is the guess?
        if guess is None or guess=="":
            yield ("guess", -1)
        else:
            yield ("guess", log(1 + len(guess)))

class FrequencyFeature:
    def __init__(self, name):
        from buzzer import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):
        import json
        if 'json.gz' in question_source:
            with gzip.open(question_source) as infile:
                questions = json.load(infile)
        else:
            with open(question_source) as infile:
                questions = json.load(infile)
        for ii in questions:
            self.counts[self.normalize(ii["page"])] += 1

    def __call__(self, question, run, guess):
        yield ("guess", log(1 + self.counts[self.normalize(guess)]))

class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """
    def __call__(self, question, run, guess):
        yield ('true', len(guess) == 0)


class GuessCapitals(Feature):
    """
    Capital letters in guess
    """
    def __call__(self, question, run, guess):
        yield ('true', log(sum(i.isupper() for i in guess) + 1))

def clean(s):
    return s.replace('&quot;', '\"') \
            .replace('\n', '') \
            .replace('\r', '') \
            .replace('_', ' ')

"""
Feature set based on Wikipedia info
"""
class WikipediaFeature(Feature):
    def __init__(self, name):
        from buzzer import normalize_answer
        self.normalize = normalize_answer
        self.name = name
        from sklearn.feature_extraction.text import TfidfVectorizer
        with open('./data/wiki_page_text.json', 'r') as f:
        # with open('/home/neal/nlp-hw/feateng/data/wiki_page_text.json', 'r') as f:
            wiki = json.load(f)
        self.vec = TfidfVectorizer()
        wiki = [p for p in wiki if p is not None and p['page'] is not None and p['text'] is not None]
        self.doc_names = [self.normalize(clean(p['page'])) for p in wiki]
        # self.docs = self.vec.fit_transform([clean(p['text']) for p in wiki])

        # self.matches = 10

    def __call__(self, question, run, guess):
        norm_guess = self.normalize(guess)

        # is guess in wiki
        in_wiki = int(norm_guess in self.doc_names)
        yield ('in_wiki', int(norm_guess in self.doc_names))
        
        tf_q = self.vec.transform([question['first_sentence']])
        if in_wiki:
            # similarity between guess and wiki page
            idx = self.doc_names.index(norm_guess)
            a = tf_q
            b = self.docs[idx]
            sim = linear_kernel(tf_q, self.docs[idx])[0][0]
            yield ('sim', sim)
        else:
            yield ('sim', 0)
        # get question first sentence
        tf_q = self.vec.transform([question['first_sentence']])

        # get similar Wikipedia sentences by tfidf
        cos_sim = linear_kernel(tf_q, self.docs).flatten()
        best_match_idx = cos_sim.argsort()[:-self.matches:-1]

        related_sim = cos_sim[best_match_idx]
        related_labels = [self.doc_names[i] for i in best_match_idx]
        guess_in_rel = [norm_guess in s for s in related_labels]
        if any(guess_in_rel) and len(guess) > 0:
            guess_idx = guess_in_rel.index(True)

            # tfidf rank and similarity of doc from guess in relation to first sentence, 0 if guess did not appear
            yield('guess_rank_rel_fs', guess_idx)

            yield('guess_sim_rel_fs', related_sim[guess_idx])
        else:
            yield('guess_rank_rel_fs', 100)
            yield('guess_sim_rel_fs', 0)
        pass

"""
TFIDF-based features
"""
class TfidfFeature(Feature):
    def __init__(self, name):
        import pickle
        self.name = name
        with open('./data/tfidf_featurizer.pkl', 'rb') as infile:
            self.featurizer = pickle.load(infile)
        with open('./data/tfidf_sentences.pkl', 'rb') as infile:
            self.wiki_sentences = pickle.load(infile)
        with open('./data/tfidf_summary_cat_link.pkl', 'rb') as infile:
            self.wiki_summaries = pickle.load(infile)

        self.matches = 20

    def __call__(self, question, run, guess):
        # get question first sentence
        tf_q = self.featurizer.transform([question['first_sentence']])

        # get similar Wikipedia sentences by tfidf
        cos_sim = linear_kernel(tf_q, self.wiki_sentences['tfidf']).flatten()
        best_match_sentences_idx = cos_sim.argsort()[:-self.matches:-1]

        related_sentences_sim = cos_sim[best_match_sentences_idx]
        related_sentences_labels = [self.wiki_sentences['labels'][i] for i in best_match_sentences_idx]
        guess_in_rel = [guess in s for s in related_sentences_labels]
        if any(guess_in_rel) and len(guess) > 0:
            guess_idx = guess_in_rel.index(True)

            # tfidf rank and similarity of sentence from guess in relation to first sentence, 0 if guess did not appear
            yield('guess_rank_rel_fs', guess_idx)

            #yield('guess_sim_rel_fs', related_sentences_sim[guess_idx])
        else:
            yield('guess_rank_rel_fs', 0)
            # yield('guess_sim_rel_fs', 0)
        pass

# enumerate numpy array
def enum_arr(arr, name):
    return [("%s_%d" % (name, i), e) for i, e in enumerate(arr)]

"""
Text vectorizer features
"""
class TextVectorizerFeature(Feature):
    """
    Abstract class for feature that produces an (SKLearn) vectorization of text
    """

    def __init__(self, name):
        self.name = name

    def add_training(self, question_source, vectorizer):
        print('training')
        corpus = get_question_field(question_source, 'text')
        self.fit_vec = vectorizer.fit(corpus)

    def vectorize(self, text, text_name):
        return enum_arr(self.fit_vec.transform([text]).toarray()[0], text_name)

class CountsVecFeature(TextVectorizerFeature):
    """
    Feature that gets count vectorized question text
    """
    def add_training(self, question_source):
        from sklearn.feature_extraction.text import CountVectorizer
        super().add_training(question_source, CountVectorizer(max_features=100))

    def __call__(self, question, run, guess):
        yield from super().vectorize(question['first_sentence'], "first_sentence")
        yield from super().vectorize(run, "run")
        yield from super().vectorize(guess, "guess")

class TfidfVecFeature(TextVectorizerFeature):
    """
    Feature that gets TF-IDF vectorized question text
    """
    def add_training(self, question_source):
        from sklearn.feature_extraction.text import TfidfVectorizer
        super().add_training(question_source, TfidfVectorizer(max_features=100))

    def __call__(self, question, run, guess):
        yield from super().vectorize(question['first_sentence'], "first_sentence")
        yield from super().vectorize(run, "run")
        yield from super().vectorize(guess, "guess")

"""
Whether guess is an English phrase (i.e., not a name or foreign phrase)
"""
class WordnetFeature(Feature):
    def __call__(self, question, run, guess):
        toks = guess.split(' ')
        r = [wordnet.synsets(t) for t in toks]
        yield ('true', bool(question['gameplay']))
