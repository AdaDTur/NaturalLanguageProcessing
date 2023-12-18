'''
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET
import codecs
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import ast
from nltk.corpus import brown
import gensim
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import gensim.downloader
from gensim.test.utils import common_texts
import gensim.downloader as api

"""
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("reuters")
"""

class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id  # id of the WSD instance
        self.lemma = lemma  # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index  # index of lemma within the context

    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)


def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances


def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        # print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key


def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}


    # read to use here
    #print(len(dev_instances))  # number of dev instances
    #print(len(test_instances))  # number of test instances

    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    examples = open("examples.txt", "w")
    vectorizer = CountVectorizer(stop_words='english')
    model = api.load("glove-twitter-25")

    #FIRST METHOD
    def compute_lesk():
        corr = 0
        counter = 0
        for x in test_instances.keys():
            sent = test_instances[x].context
            word = test_instances[x].lemma
            word = word.decode("UTF-8")
            idn = test_instances[x].id
            correct_sens = test_key[idn]
            stopped = [w.decode("UTF-8") for w in sent if not w.lower() in stop_words]
            lemmatized = [lemmatizer.lemmatize(word) for word in stopped]
            res = nltk.wsd.lesk(lemmatized, word, "n")
            conv = str(res.lemmas()[0].key())
            if conv in correct_sens:
                corr += 1
            counter += 1
        accuracy = corr/counter
        print("Accuracy: " + str(accuracy))
        return accuracy

    #SECOND METHOD
    def compute_most_freq():
        counter = 0
        freq_corr = 0
        for x in test_instances.keys():
            word = test_instances[x].lemma.decode("UTF-8")
            idn = test_instances[x].id
            correct_sens = test_key[idn]
            pred = wn.synsets(word)[0]
            pred_conv = str(pred.lemmas()[0].key())
            if pred_conv not in correct_sens:
                freq_corr += 1
            counter += 1
        accuracy = freq_corr/counter
        print("Accuracy: " + str(accuracy))
        return accuracy

    #THIRD METHOD

    def run_nb(input_word):
        def get_examples(input_word):
            ex_dict = {}
            word = input_word
            for sense_id in range(len(wn.synsets(word))):
                if sense_id >= 4: #limiting to four most frequent senses
                    break
                ex_dict[word + "_" + str(sense_id)] = wn.synsets(word)[sense_id].examples()
            return ex_dict

        def make_test(input_word):
            test_set = []
            test_vals = []
            for x in test_instances.keys():
                lemma = test_instances[x].lemma
                lemma_dec = wn.synset_from_sense_key(test_key[test_instances[x].id][0]).name().split('.')[0]
                if lemma_dec == input_word:
                    sent = test_instances[x].context
                    decoded = [w.decode("UTF-8").lower() for w in sent]
                    joined = ' '.join(decoded)
                    test_set.append(joined)

                    idn = test_instances[x].id
                    correct_sens = wn.synset_from_sense_key(test_key[idn][0]).name()
                    test_vals.append(correct_sens)
            return test_set, test_vals

        examples = get_examples(input_word)
        x = []
        y = []
        return_acc = []
        for n in range(len(examples.values())):
            sense = list(examples.keys())[n]
            for sent in list(examples.values())[n]:
                x.append(sent)
                y.append(sense)
        brown_sents = brown.sents()
        brown_join = [' '.join([lemmatizer.lemmatize(n.lower()) for n in w]) for w in brown_sents if (input_word in w)]
        train_size = len(x)
        test_set, test_vals = make_test(input_word)
        test_size = len(test_set)
        all = test_set + x + brown_join
        all_vec = vectorizer.fit_transform(all)
        for i in range(3):
            classifier = MultinomialNB(alpha=0.4)
            classifier.fit(all_vec[test_size:train_size + test_size], y)
            test_pred = classifier.predict(all_vec[:test_size])
            test_pred_keys = []
            for w in test_pred:
                w_split = w.split("_")
                targ_word = w_split[0]
                targ_num = int(w_split[1])
                test_pred_keys.append(wn.synset_from_sense_key(wn.synsets(targ_word)[targ_num].lemmas()[0].key()).name())
            accuracy = accuracy_score(test_vals, test_pred_keys)
            #print(f"Accuracy: {accuracy}")
            return_acc.append(accuracy)
            y_pred_probs = classifier.predict_proba(all_vec[train_size + test_size:])
            y_pred = classifier.predict(all_vec[train_size + test_size:])
            og_train_size = train_size
            num = {}
            for n in range(len(y_pred_probs)):
                if max(y_pred_probs[n]) > 0.6 and y_pred[n] not in num: #limiting to only adding one new example to train set for each sense, and by confidence threshold
                    num[y_pred[n]] = True
                    index = og_train_size + test_size + n
                    tmp = all[index]
                    all[index] = all[train_size + test_size]
                    all[train_size + test_size] = tmp
                    y.append(y_pred[n])
                    train_size += 1
        return return_acc

    def yarowsky():
        words = ["claim", "degree", "state", "time", "dinner", "end"]
        for word in words:
            res = run_nb(word)
            print(word, res)

    #FOURTH METHOD
    def cos_sim(vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot / (norm1 * norm2)
        return similarity


    def word_vec(sentence):
        if len(sentence) > 0:
            length = 0
            stopped = [w for w in sentence.split() if w not in stop_words]
            res = np.array(np.zeros(25,))
            for word in stopped:
                if word in model:
                    length += 1
                    vec = model[word.lower()]
                    res = res + vec
                else:
                    continue
            res = res / length
            return res
        else:
            return np.array(np.zeros(25,))


    def embed_method():
        count = 0
        corr = 0
        for n in test_instances:
            idn = test_instances[n].id
            sent = ' '.join(w.decode("UTF-8") for w in test_instances[n].context)
            word = test_instances[n].lemma
            word = word.decode("UTF-8")
            vec1 = word_vec(sent) #vectorization of context
            max = 0
            val = wn.synsets('book')[0] #setting to some dummy value, since we know it will be changed
            for w in wn.synsets(word):
                tmp = w.lemmas()[0].key()
                if tmp in [h[0] for h in test_key.values()]: #if the sense is in the dev set senses
                    vec2 = word_vec(w.definition())
                    sim = cos_sim(vec1, vec2) #getting sense with highest similarity to context on basis of definition
                    if sim > max:
                        max = sim
                        val = w
            if val.lemmas()[0].key() == test_key[idn][0]:
                corr += 1
            count += 1
        accuracy = corr/count
        print("Accuracy: " + str(accuracy))
        return accuracy

    #First approach
    print("====RUNNING FIRST METHOD====")
    compute_most_freq()

    #Second approach
    print("====RUNNING SECOND METHOD====")
    compute_lesk()

    #Third approach
    print("====RUNNING THIRD METHOD====")
    yarowsky()

    #Fourth approach
    print("====RUNNING FOURTH METHOD====")
    embed_method()

