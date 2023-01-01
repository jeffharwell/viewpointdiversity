import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Cleaning
import contractions
from string import punctuation

# Topic Modeling
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore


class TopicFeatureGenerator:
    """
    Takes raw texts and generates a topic feature vector for each text.
    Also allows you to retrieve the learned topic model. Uses Gensim for the
    topic modeling and includes bigrams.
    """

    def __init__(self):
        # Initialize the lemmatizer
        self.wnl = WordNetLemmatizer()

        # Get our list of stopwords
        stop_words = set(stopwords.words("english"))
        self.stop_words = [s for s in stop_words if s not in ['no', 'nor', 'not']]  # I want negations

        # Constants for the LDA model, your mileage may vary
        self.chunksize = 1000
        self.passes = 40
        self.iterations = 400
        self.workers = 7

        self.lda_model = None
        self.num_topics = None
        self.coherence_score = None
        self.topic_vectors = None
        self.topics = None
        self.debug = False
        self.min_number_topics = 4
        self.max_number_topics = 15

    def create_topic_vectors_from_texts(self, texts):
        """
        Create topic vectors for each text in texts

        :param texts: a list of texts
        :return: a list of topic vectors (list of lists of floats)
        """

        # preprocess and clean the text
        normalized_texts = [self.clean_text(t) for t in texts]
        if self.debug:
            print("Document 0 normalized")
            print(normalized_texts[0])
        # get text with bigram phrases
        texts_with_phrases = self.text_with_bigram_phrases(normalized_texts)
        if self.debug:
            print("Document 0 text with bigram phrases")
            print(texts_with_phrases[0])
        # Build a frequency dictionary and filter out the very rare words
        id2word = Dictionary(texts_with_phrases)
        id2word.filter_extremes(no_below=.01, no_above=0.5)
        # create the BOW representation using the dictionary
        bow_corpus = [id2word.doc2bow(doc) for doc in texts_with_phrases]
        if self.debug:
            print("BOW Corpus Document 0")
            print(bow_corpus[0])
            tokens_in_first_corpus = [id2word[t[0]] for t in bow_corpus[0]]
            print("Tokens in BOW Corpus 0")
            print(" ".join(tokens_in_first_corpus))
        # get the optimum number of topics
        num_topics, coherence_scores = self.find_best_num_topics(self.min_number_topics, self.max_number_topics,
                                                                 id2word, bow_corpus, texts_with_phrases)
        if self.debug:
            tuples = [str(t) for t in coherence_scores]
            print("Number of Topics, Coherence Numbers")
            print(" ".join(tuples))

        # train a model based on the number of topics that maximizes coherence
        lda_model = self.get_lda_model(num_topics, id2word, bow_corpus)
        # get the coherence for the model
        coherence_score = self.measure_coherence(num_topics, id2word, bow_corpus, texts_with_phrases)
        # create the tokens describing each topic
        topics = list()
        for topic_id in range(0, num_topics):
            topics.append(" ".join([id2word[tid[0]] for tid in lda_model.get_topic_terms(topic_id)]))
        topic_vectors = self.create_topic_vectors(num_topics, lda_model, bow_corpus)

        self.lda_model = lda_model
        self.num_topics = num_topics
        self.topic_vectors = topic_vectors
        self.topics = topics
        self.coherence_score = coherence_score

    def create_topic_vectors(self, num_topics, lda_model, bow_corpus):
        """
        Create a topic vector for each document in the corpus.

        model.get_document_topics(bow_document_rep) looks like the following:
        [(1, 0.28715208), (4, 0.20299017), (6, 0.48378795)]
        we want to turn that into a list representing the document's representation
        of each topic, with a zero for topics that it does not represent at all.

        :param num_topics: number of topics in the model
        :param lda_model: the LDA model
        :param bow_corpus: the Gensim BOW corpus
        :return: a list of lists of floats representing the contribution of each topic to the text
        """

        topic_vectors = list()  # the list that will hold all of our topic representations
        topic_index = list(range(0, num_topics))  # a list with the index of each topic

        for c in bow_corpus:  # each text represented as a Gensim BOW
            topic_dict = {}  # convert the list of tuples into a dictionary of topics for this document
            topic_list = []
            for tup in lda_model.get_document_topics(c):  # get our topics
                topic_dict[tup[0]] = tup[1]  # save them to the dictionary
            for t in topic_index:  # go through each topic
                if t in topic_dict:
                    # we have a value for that topic, append it
                    topic_list.append(topic_dict[t])
                else:
                    # we don't have a value for this topic, so it will be 0.0
                    topic_list.append(0.0)
            topic_vectors.append(topic_list)  # add the topic vector we just created to the list of all topic vectors
        return topic_vectors  # return our list of all topic vectors

    def find_best_num_topics(self, min_num_topics, max_num_topics, id2word, bow_corpus, texts):
        """
        Get the number of topics that gives the best coherence for the LDA model

        :param min_num_topics: minimum number of topics
        :param max_num_topics: maximum number of topics
        :param id2word: the gensim id2word dictionary
        :param bow_corpus: a gensim BOW corpus
        :param texts: list of tokenized texts with bigrapm phrases over which to evaluate coherence
        :return: the optimal number of topics, list of tuples with (num topics, coherence score)
        """
        max_coherence = 0
        num_topics = min_num_topics
        all_coherence_scores = list()

        for i in range(min_num_topics, max_num_topics):
            coherence = self.measure_coherence(i, id2word, bow_corpus, texts)
            all_coherence_scores.append((i, coherence))
            if coherence > max_coherence:
                max_coherence = coherence
                num_topics = i

        return num_topics, all_coherence_scores

    def get_lda_model(self, num_topics, id2word, bow_corpus):
        """
        Return an LDA model trained on the corpus for a given number of topics.

        :param num_topics: the number of topics to use
        :param id2word: the gensim id2word dictionary
        :param bow_corpus: a gensim BOW corpus
        :return: an Gensim LDA model
        """
        lda_model = LdaMulticore(workers=self.workers, corpus=bow_corpus,
                                 # lda_model = LdaModel(corpus = corpus,
                                 id2word=id2word,
                                 num_topics=num_topics,
                                 random_state=100,
                                 chunksize=self.chunksize,
                                 passes=self.passes,
                                 # alpha = 'auto',
                                 eta='auto',
                                 iterations=self.iterations,
                                 per_word_topics=True,
                                 eval_every=None)
        return lda_model

    def measure_coherence(self, num_topics, id2word, bow_corpus, texts):
        """
        Measures the coherence of a LDA model with a given set of topics over a set of texts.

        :param num_topics: the number of topics to use
        :param id2word: the gensim id2word dictionary
        :param bow_corpus: a gensim BOW corpus
        :param texts: list of tokenized texts with bigrapm phrases over which to evaluate coherence
        :return:
        """

        lda_model = self.get_lda_model(num_topics, id2word, bow_corpus)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda

    def text_with_bigram_phrases(self, texts):
        """
        Create a text containing the bigraam phrases

        :param texts: list of normalized texts
        :return: texts as a list of lists of tokens containing unigram and bigram phrases
        """
        sent = [text.split() for text in texts]
        bigram = Phrases(sent, min_count=25)
        bigram_phraser = Phraser(bigram)
        texts_w_phrases = bigram_phraser[sent]

        return texts_w_phrases

    def clean_text(self, text):
        """
        Takes a text and cleans and normalizes it. This includes:

        1. contract expansion
        2. lemmatization
        3. lowercase
        4. removing non-alphanumeric tokens
        5. removing punctuation
        6. removing tokens that are less than 3 characters long

        :param text: a string to normalize
        :return: a normalized string of text
        """
        expanded_text = self.contract_expansion(text)
        lemmas = self.lemmatize(self.pos_tag(expanded_text))
        tokens = [token.lower() for token in lemmas]
        tokens = [token for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in punctuation]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [token for token in tokens if len(token) >= 2]
        return ' '.join(tokens)

    def contract_expansion(self, input_string):
        """
        Expand all contractions in the given input string

        :param input_string: the text to expand
        :return: text with contractions expanded
        """
        expanded_word = []
        for word in input_string.split():
            expanded_word.append(contractions.fix(word))
        return ' '.join(expanded_word)

    def pos_tag(self, input_string):
        """
        Tag each word in the input string after tokenizing it using NLTK wordpunct
        tokenizer. This using the NLTK POS tagging library. The output is a list
        of tuples with the token and the tag, like the below:

        [('this', 'DT'),
         ('is', 'VBZ'),
         ('my', 'PRP$'),
         ('first', 'JJ'),
         ('sentence', 'NN'),
         ('.', '.')]

        :param input_string: The text to tag
        :return: list of tuples of tokens and tags
        """
        words = nltk.wordpunct_tokenize(input_string)  # tokenize
        tagged = nltk.pos_tag(words)  # tag
        return tagged

    def lemmatize(self, tagged):
        """
        Use wordnet to lemmitize the text based on POS tags. Expects NLTK POS tags

        :param tagged: list of token, tag tuples: [('this', 'DT'), ('is', 'VBZ')]
        :return: list of lemmatized tokens
        """
        lemmas = []
        for word, tag in tagged:
            if tag.startswith('NN'):
                token = self.wnl.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                token = self.wnl.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                token = self.wnl.lemmatize(word, pos='a')
            elif tag.startswith('RB'):
                token = self.wnl.lemmatize(word, pos='r')
            else:
                token = word
            lemmas.append(token)
        return lemmas



