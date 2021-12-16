"""
Module containing various functions I have found useful in assembling the feature vector.
"""

import numpy as np
from viewpointdiversitydetection.TokenFilter import TokenFilter


class GenerateSentencesAndContexts:
    def __init__(self, matching_term, context_size, stemmer):
        self.matching_term = matching_term
        self.context_size = context_size
        self.token_filter = TokenFilter()
        self.stemmer = stemmer

    def generate(self, parsed_document):
        sac = SentencesAndContexts(self.matching_term, self.context_size,
                                   parsed_document, self.stemmer, self.token_filter)
        return sac


class SentencesAndContexts:
    def __init__(self, matching_term, context_size, parsed_document, stemmer, token_filter):
        self.matching_term = stemmer.stem(matching_term)
        self.context_size = context_size

        self.contexts = list()
        self.context_boundaries = list()
        self.sentences = list()
        self._extract(parsed_document, stemmer, token_filter)

    def _get_context_tokens(self, token_list, term_index):
        context_start = term_index - self.context_size
        if context_start < 0:
            context_start = 0
        context_end = term_index + self.context_size + 1  # +1 because slices are exclusive of endpoint
        if context_end > len(token_list):
            context_end = len(token_list)
        context = token_list[context_start: context_end]
        return context, context_start, context_end

    def _extract(self, parsed_document, stemmer, tf):
        for s in parsed_document.sents:  # iterate through each sentence in the parsed document
            raw_tokens = [str(t) for t in s if tf.filter(t)]  # convert the sentence <span> into tokens
            # Here! I need to keep track of the original index of each
            # token so that I can state the start and end of any
            # extracted contexts in terms of the full sentence, not
            # just in terms of the filtered token set.
            stemmed_tokens = [stemmer.stem(t) for t in raw_tokens]  # convert the sentence <span> into tokens
            if self.matching_term in stemmed_tokens:  # see if we have a matching term
                term_index = stemmed_tokens.index(self.matching_term)  # note getting matching term using the stemmed
                context, context_start, context_end = self._get_context_tokens(raw_tokens, term_index)

                self.sentences.append(s)
                self.contexts.append(context)
                self.context_boundaries.append((context_start, context_end))


def compute_avg_negation_difference(sent_and_contexts1, sent_and_contexts2):
    """
    We compare each context in document 1 to each context in document 2
    the difference in negations is abs(# negations in context 1 - # negations in context 2)

    Summing up the differences in negations between all the contexts and dividing by the
    total number of combinations gives us the average negation difference between
    the two documents.

    Note that this might need to be modified by the Word Mover Distance (WMD). If two sentences have a
    low WMD then a difference in negation likely means that there is a distinct difference in meaning, however, if the
    two sentences have a relatively large WMD then the effect of the negations is less defined in terms of meaning.

    Currently filtering the entities by context is broken because the context boundaries coming out of the
    SentencesAndContexts object are posting token filtering, but the entity start and end are pre token filtering,
    and currently I don't have a way to reconcile those. I'd like to compute the start and end of the context
    pre-filtering in the SentenceAndContexts object but I'm not quite sure how to do it robustly yet. If this
    methodology shows promise I will figure that part out.

    :param sent_and_contexts1: the SentenceAndContexts object from document 1
    :param sent_and_contexts2: the SentenceAndContexts object from document 2
    :return: the average difference in negations between the documents as a float
    """
    differences_in_negation = list()
    for ci1, sent1 in zip(sent_and_contexts1.context_boundaries, sent_and_contexts1.sentences):
        for ci2, sent2 in zip(sent_and_contexts2.context_boundaries, sent_and_contexts2.sentences):
            n1 = get_number_of_negations(sent1, 0, len(sent1))
            n2 = get_number_of_negations(sent2, 0, len(sent2))
            # print(sent1, ci1[0], ci1[1])
            # print(sent2, ci2[0], ci2[1])
            # if n1 > 0:
            #     print("Found negations in context 1")
            #  if n2 > 0:
            #     print("Found negations in context 2")
            differences_in_negation.append(abs(n1-n2))

    if len(differences_in_negation) > 0:
        avg_negation_difference = sum(differences_in_negation) / len(differences_in_negation)
    else:
        avg_negation_difference = 0.0

    return avg_negation_difference


def compute_avg_wmd_distance(contexts1, contexts2, gensim_obj):
    """
    Compute an average word metric distance
    :param contexts1: List of strings representing contexts extracted from Document 1
    :param contexts2: List of strings representing contexts extracted from Document 2
    :param gensim_obj: a Gensim object initialized with a word2vec model
                       e.g. gensim_obj = api.load('word2vec-google-news-300')
    :return: Dictionary with keys for the 'max' WMD, 'min' WMD, and 'avg' WMD for all
             combinations of contexts. Returns None if the average WMD cannot be calculated
    """
    distances = list()
    context_combinations = 0

    if len(contexts1) == 0:
        return None
    if len(contexts2) == 0:
        return None

    # Calculate the distance for every combination of contexts we have been passed
    for c1 in contexts1:
        for c2 in contexts2:
            sc1 = " ".join(c1)
            sc2 = " ".join(c2)
            context_combinations += 1
            distances.append(gensim_obj.wmdistance(sc1, sc2))

    if len(distances) > 0:
        avg_distance = sum(distances) / len(distances)
    else:
        return None

    max_distance = max(distances)
    min_distance = min(distances)

    return {'max': max_distance, 'min': min_distance, 'avg': avg_distance,
            'number_of_combinations': context_combinations}


def get_number_of_negations(space_sent_obj, context_start, context_end):
    """
    Calculates the number of negated entities in the spacy sentence
    :param space_sent_obj: spacy sentence span
    :param context_start: the index of the token marking the beginning of the context we are considering
    :param context_end: the index of the token marking the end of the context we are considering
    :return: number of negated entities in the span
    """
    number_of_negations = 0
    for e in space_sent_obj.ents:
        if e._.negex:
            # Consider context. If the entity we are examining overlaps the context boundaries we count
            # it anyways
            # Note that since the end of the context and the end of the entities is exclusive,
            # so < and > not <= and >=
            if e.start < context_end and e.end > context_start:
                number_of_negations += 1
    return number_of_negations


def create_document_index_combinations(pdo):
    """
    Compute all possible pairwise combinations of documents within the Parsed Documents

    :param pdo: a ParsedDocumentsFourForums object
    :return: all pairwise combinations of document indexes as a list of tuples
    """
    combinations = list()
    for i in range(len(pdo.all_docs)-1):
        doc_id_1 = i
        for j in range(i + 1, len(pdo.all_docs)):
            doc_id_2 = j
            combinations.append((doc_id_1, doc_id_2))
    return combinations
