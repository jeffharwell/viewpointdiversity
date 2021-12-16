import math
import re
import unittest
import configparser

from nltk import SnowballStemmer

from src.viewpointdiversitydetection.compare_documents import compute_avg_wmd_distance
from src.viewpointdiversitydetection.compare_documents import create_document_index_combinations
from src.viewpointdiversitydetection.compare_documents import get_number_of_negations
from src.viewpointdiversitydetection.compare_documents import compute_avg_negation_difference
from src.viewpointdiversitydetection.compare_documents import GenerateSentencesAndContexts
from viewpointdiversitydetection import ParsedDocumentsFourForums
from viewpointdiversitydetection import TokenFilter
import gensim.downloader as api

#from negspacy.negation import Negex
from negspacy.termsets import termset
import spacy
import torch

class CompareDocumentsTest(unittest.TestCase):
    """
    A few tests for the ParsedDocumentsFourForums class. These some of these assume that you have a connection
    to a copy of the Internet Argument Corpus database. Put the database host, username, and password in the
    config.ini file before testing.
    """
    def setUp(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.tf = TokenFilter()

        self.user = config['InternetArgumentCorpus']['username']
        self.password = config['InternetArgumentCorpus']['password']
        self.host = config['InternetArgumentCorpus']['host']
        self.database = 'fourforums'

    def test_combinations(self):
        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 40
        pdo.set_result_limit(limit)
        pdo.stance_agreement_cutoff = 1  # disable the inter-annotator cutoff
        pdo.extract_negations = True
        pdo.spacy_model = 'en_core_web_trf'
        pdo.process_corpus()

        combinations = create_document_index_combinations(pdo)
        number_of_pairs = (len(pdo.all_docs) * (len(pdo.all_docs) - 1)) / 2  # combination formula for r = 2
        # Make sure we have the right number of pairs
        self.assertTrue(len(combinations) == number_of_pairs)
        # Our pairs should always be different ID numbers
        for c in combinations:
            self.assertTrue(c[0] != c[1])

    def test_compute_avg_wmd(self):
        """
        Test function that computes the average, min, and maximum WMD of the context extracted
        from pairs of documents.
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 40
        pdo.set_result_limit(limit)
        pdo.stance_agreement_cutoff = 1  # disable the inter-annotator cutoff
        pdo.extract_negations = True
        pdo.spacy_model = 'en_core_web_trf'
        vector_model = api.load('glove-wiki-gigaword-50')
        pdo.process_corpus()
        stemmer = SnowballStemmer(language='english')

        context_generator = GenerateSentencesAndContexts('gun', 6, stemmer)
        contexts_by_docid = [context_generator.generate(doc) for doc in pdo.all_docs]

        document_index_combinations = create_document_index_combinations(pdo)
        wmds = [compute_avg_wmd_distance(contexts_by_docid[i[0]].contexts, contexts_by_docid[i[1]].contexts,
                                         vector_model) for i in document_index_combinations]

        # We should get at least a few valid comparisons for 40 documents and a term as common as 'gun' in this corpus
        number_of_wmd_values = sum([1 for w in wmds if w])
        self.assertTrue(number_of_wmd_values > 0)

        # The values we get should follow the below rules
        for pair, wmd in zip(document_index_combinations, wmds):
            if wmd:
                if wmd['number_of_combinations'] == 1:
                    self.assertTrue(wmd['max'] == wmd['min'])
                    self.assertTrue(wmd['max'] == wmd['avg'])
                if wmd['number_of_combinations'] > 1:
                    self.assertTrue(wmd['max'] >= wmd['min'])
                    self.assertTrue(wmd['max'] >= wmd['avg'])
                    self.assertTrue(wmd['avg'] >= wmd['min'])

    def test_calculate_number_of_negations(self):
        # torch.set_num_threads(1)
        ts = termset("en")
        nlp = spacy.load('en_core_web_trf')
        nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})

        doc1 = nlp("It is not Los Angeles.")
        doc2 = nlp("It is Los Angeles.")
        doc3 = nlp("It is not Los Angeles and it is not Chicago")

        doc1_negations = get_number_of_negations(next(doc1.sents), 0, 5)
        self.assertTrue(doc1_negations == 1)

        doc2_negations = get_number_of_negations(next(doc2.sents), 0, 4)
        self.assertTrue(doc2_negations == 0)

        doc3_sent = next(doc3.sents)

        doc3_negations = get_number_of_negations(doc3_sent, 0, 10)
        self.assertTrue(doc3_negations == 2)

        # There is only one negation within this context
        doc3_c2_negations = get_number_of_negations(doc3_sent, 0, 5)
        self.assertTrue(doc3_c2_negations == 1)

        # There are no negations within this context
        doc3_c3_negations = get_number_of_negations(doc3_sent, 0, 2)
        self.assertTrue(doc3_c3_negations == 0)

        # If the negated entity overlaps with the boundaries of the context
        # we still count it as a negation in the context
        doc3_c4_negations = get_number_of_negations(doc3_sent, 0, 4)
        self.assertTrue(doc3_c4_negations == 1)

    def test_average_negation_difference(self):
        """
        Test our computation of the average negation difference between documents
        """
        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 40
        pdo.set_result_limit(limit)
        pdo.stance_agreement_cutoff = 1  # disable the inter-annotator cutoff
        pdo.extract_negations = True
        pdo.spacy_model = 'en_core_web_trf'
        # vector_model = api.load('glove-wiki-gigaword-50')
        pdo.process_corpus()
        stemmer = SnowballStemmer(language='english')

        context_generator = GenerateSentencesAndContexts('gun', 6, stemmer)
        contexts_by_docid = [context_generator.generate(doc) for doc in pdo.all_docs]

        document_index_combinations = create_document_index_combinations(pdo)

        negation_difference = [compute_avg_negation_difference(contexts_by_docid[i[0]], contexts_by_docid[i[1]])
                               for i in document_index_combinations]

        # print(negation_difference)
        self.assertTrue(sum(negation_difference) > 0.0)  # we should find at least 1 thing ...


if __name__ == '__main__':
    unittest.main()
