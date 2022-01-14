"""
Module containing various functions I have found useful in assembling the feature vector.
"""
import hashlib

# import numpy as np
# import spacy

from viewpointdiversitydetection.TokenFilter import TokenFilter


class GenerateSentencesAndContexts:
    def __init__(self, matching_term, context_size, stemmer):
        self.matching_term = matching_term
        self.context_size = context_size
        self.token_filter = TokenFilter()
        self.stemmer = stemmer

    def generate(self, parsed_document, return_spans=False):
        sac = SentencesAndContexts(self.matching_term, self.context_size,
                                   parsed_document, self.stemmer, self.token_filter)
        if return_spans:
            spans = sac.extract(return_spans=True)
            return spans, sac
        else:
            sac.extract()
            return sac


class ContextAndEntities:
    def __init__(self, sentence_span, context_tokens, context_start, context_end, entities_list,
                 entities_start_list, entities_end_list, entities_are_negated_list):
        """
        Class which holds one context, the sentence the context came from, any entities identified in that sentence
        and the negation status of those entities.

        :param sentence_span: a Spacy Span object containing a sentence
        :param context_tokens: the extracted context as a list of tokens
        :param context_start: the index, relative to the full sentence, where the context begins (inclusive)
        :param context_end: the index, relative to the full sentence, where the context ends (exclusive)
        :param entities_list: the list of identified entities
        :param entities_start_list: a list of start indices for the entities
        :param entities_end_list: a list of ending indices for the entities
        :param entities_are_negated_list: the negation status of each entity
        """

        self.sentence_as_string = str(sentence_span)
        self.sentence_as_tokens = [t.text for t in sentence_span]
        self.context_tokens = context_tokens
        self.context_start = context_start
        self.context_end = context_end
        self.entities_list = entities_list
        self.entities_start = entities_start_list
        self.entities_end = entities_end_list
        self.entities_are_negated_list = entities_are_negated_list

    def __str__(self):
        """
        This class is primarily centered on the context, so return the context as a string.

        :return: extracted context as a string.
        """
        return " ".join(self.context_tokens)

    def get_number_of_negations(self):
        """
        Calculates the number of negated entities within this context
        """

        number_of_negations = 0
        for index, e in enumerate(self.entities_list):
            if self.entities_are_negated_list[index]:
                # print(f"Found a negation for entity {e}")
                # print(f"looking in first sentence: {self.context_start} to {self.context_end}")

                # Consider context. If the entity we are examining overlaps the context boundaries we count
                # it anyways
                # Note that since the end of the context and the end of the entities is exclusive,
                # so < and > not <= and >=
                if self.entities_start[index] < self.context_end and self.entities_end[index] > self.context_start:
                    # print(f"Negation is within the context: {self.sentence_as_string}")
                    number_of_negations += 1
        return number_of_negations


class SentencesAndContexts:
    """
    Class which holds all the contexts from a Spacy parsed document that match a specific term. Because Spacy
    Span objects don't pickle the class does not directly store any Spacy Spans, it only stores Spacy Docs and
    information derived from them. Interesting, and frustratingly enough, spacy.Span.as_doc() keeps entities but
    looses any negation information added by Negex, so entity and negation information is saved separately in the
    SentenceWithEntities class and the various methods in this module that deal with negations do not try to pull
    the negation information directly from the Spacy documents.
    """
    def __init__(self, search_term, context_size, parsed_document, stemmer, token_filter):
        self.parsed_document = parsed_document
        self.stemmer = stemmer
        self.token_filter = token_filter

        self.original_search_term = search_term
        self.stemmed_search_term = stemmer.stem(search_term)
        self.context_size = context_size
        self.contexts_and_entities = list()

    def _get_context_tokens(self, token_list, filtered_token_list, term_index, original_indices):
        """
        This is a bit tricky because we want to return the start and end of the context in terms of the original
        token list, not the filtered token list. So first calculate the context size in terms of the filtered token
        list, and then use the list which maps the filtered tokens to their original indices to calculate the
        start and end of the context in terms of the original sentence.

        :param token_list: a list of tokens that comprises the sentence, list of strings
        :param filtered_token_list: the filtered list of tokens used to match the context
        :param term_index: the index of the matching term
        :param original_indices: list mapping each element in the filtered_token_list to its original index in
                                 the token_list
        :return: a list of tokens comprising the context, the index the contexts starts, and the index the context ends.
                 note that the context end is exclusive in keeping with the Python list slicing conventions.
        """
        context_start = term_index - self.context_size
        if context_start < 0:
            context_start = 0
        original_start = original_indices[context_start]

        context_end = term_index + self.context_size + 1  # +1 because slices are exclusive of endpoint
        if context_end > len(filtered_token_list):
            context_end = len(filtered_token_list)
        original_end = original_indices[context_end - 1] + 1
        context = token_list[original_start: original_end]

        return context, original_start, original_end

    def extract(self, return_spans = False):
        """
        Do the actual extraction.

        :param return_spans: return the extracted spans, defaults to False
        """
        parsed_document = self.parsed_document
        stemmer = self.stemmer
        tf = self.token_filter
        spans = list()

        for index, s in enumerate(parsed_document.sents):  # iterate through each sentence in the parsed document
            unfiltered_tokens = [str(t) for t in s]
            filtered_tokens_and_indexes = [(i, str(t)) for i, t in enumerate(s) if tf.filter(t)]
            filtered_tokens = [i[1] for i in filtered_tokens_and_indexes]
            original_indices = [i[0] for i in filtered_tokens_and_indexes]
            stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]  # convert the sentence <span> into tokens

            # find the index of every stemmed token in the sentence that matches the stemmed matching term
            matching_indexes = [i for i, token in enumerate(stemmed_tokens) if token == self.stemmed_search_term]
            # now extract the context for every matching term, we are fine if extracted contexts overlap
            for matching_index in matching_indexes:
                context, context_start, context_end = self._get_context_tokens(unfiltered_tokens, filtered_tokens,
                                                                               matching_index, original_indices)
                # Now extract and save all entities in the sentence along with their
                # negation status
                entities_list = list()
                entities_start_list = list()
                entities_end_list = list()
                entity_is_negated_list = list()
                for e in s.ents:
                    entities_list.append(str(e))
                    entities_start_list.append(e.start)
                    entities_end_list.append(e.end)
                    if e._.negex:
                        entity_is_negated_list.append(True)
                    else:
                        entity_is_negated_list.append(False)

                spans.append(s)
                cae = ContextAndEntities(s, context, context_start, context_end, entities_list,
                                         entities_start_list, entities_end_list, entity_is_negated_list)
                self.contexts_and_entities.append(cae)

        if return_spans:
            return spans


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

    :param sent_and_contexts1: the SentenceAndContexts object from document 1
    :param sent_and_contexts2: the SentenceAndContexts object from document 2
    :return: the average difference in negations between the documents as a float
    """
    differences_in_negation = list()
    for cae1 in sent_and_contexts1.contexts_and_entities:
        for cae2 in sent_and_contexts2.contexts_and_entities:
            n1 = cae1.get_number_of_negations()
            n2 = cae2.get_number_of_negations()
            differences_in_negation.append(abs(n1-n2))

    if len(differences_in_negation) > 0:
        avg_negation_difference = sum(differences_in_negation) / len(differences_in_negation)
    else:
        avg_negation_difference = 0.0

    return avg_negation_difference


class WMDCache:
    """
    Compute the WMD distance for a pair of strings and cache that distance for future requests
    """
    def __init__(self, gensim_obj):
        self.gensim_obj = gensim_obj
        self.hash_to_wmd = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_wmd(self, s1, s2):
        """
        Get the word movers distance between two strings.

        :param s1: the first string
        :param s2: the second string
        :return: the word movers distance betwen the strings as computed by the Gensim Model
        """
        key = self._compute_key(s1, s2)
        if key in self.hash_to_wmd:
            self.cache_hits += 1
            return self.hash_to_wmd[key]
        else:
            self.cache_misses += 1
            wmd = self.gensim_obj.wmdistance(s1, s2)
            self.hash_to_wmd[key] = wmd
            return wmd

    @staticmethod
    def _compute_key(s1, s2):
        """
        Compute a key from two strings

        :param s1: string 1
        :param s2: string 2
        :return:
        """
        return hashlib.md5(s1.encode('utf-8')).hexdigest() + '-' + hashlib.md5(s2.encode('utf-8')).hexdigest()


def compute_avg_wmd_distance(contexts1, contexts2, gensim_obj):
    """
    Compute an average word metric distance
    :param contexts1: List of ContextAndEntities objects extracted from Document 1
    :param contexts2: List of ContextAndEntities objects extracted from Document 2
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
            sc1 = c1.sentence_as_string
            sc2 = c2.sentence_as_string
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


def get_number_of_negations_from_span(span_obj, context_start, context_end):
    """
    Calculates the number of negated entities in the spacy sentence
    :param span_obj: spacy span consisting of a single sentence
    :param context_start: the index of the token marking the beginning of the context we are considering
    :param context_end: the index of the token marking the end of the context we are considering
    :return: number of negated entities in the span
    """

    number_of_negations = 0
    sent_text = str(span_obj)
    for index, e in enumerate(span_obj.ents):
        # print(f"looking in first sentence: {context_start} to {context_end} entity {e}")
        if e._.negex:
            # Consider context. If the entity we are examining overlaps the context boundaries we count
            # it anyways
            # Note that since the end of the context and the end of the entities is exclusive,
            # so < and > not <= and >=
            # print(f"Found a negation for entity {e}")
            if e.start < context_end and e.end > context_start:
                # print(f"Negation is within the context: {sent_text}")
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
