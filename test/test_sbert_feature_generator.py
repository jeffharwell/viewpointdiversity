import configparser
import time
import unittest
import numpy as np

from viewpointdiversitydetection import SBertFeatureGenerator, FeatureVectorsAndTargets, RawCorpusFourForums
from viewpointdiversitydetection import TokenFilter, ParsedDocumentsFourForums, FindCharacteristicKeywords


class SBertFeatureGeneratorTest(unittest.TestCase):

    def test_avg_vector_no_divide_by_zero(self):
        """
        When all of the word2vec features are empty we shouldn't hit a divide-by-zero
        RuntimeError if we are not including zeros in the averages.

        :return:
        """
        w2v_obj = SBertFeatureGenerator(False, False)
        zero_vector = np.zeros(w2v_obj.vector_size)
        vectors = [zero_vector.copy(), zero_vector.copy(), zero_vector.copy()]
        avg = w2v_obj._average_vector(vectors)
        self.assertTrue(np.array_equal(avg, zero_vector))

        w2v_obj.include_zeros_in_averages = False

        avg = w2v_obj._average_vector(vectors)
        self.assertTrue(np.array_equal(avg, zero_vector))

    def test_create_document_embeddings(self):
        """
        Does it do anything at all, let's find out.

        :return:
        """
        """
        This is basically our end-to-end test case. Take some data from the database and create feature vectors
        from those documents. The test uses a smaller word2vec model for speed.
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(10)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        w2v_obj = SBertFeatureGenerator(False, False)
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, w2v_obj, search_terms, related_terms, context_size)
        fvt.create_feature_vectors_and_targets()
        print("First Feature Vector")
        print(fvt.feature_vectors[0])
        print("First Target Class")
        print(fvt.targets_for_features[0])
        print(f"Created {len(fvt.feature_vectors)} feature vectors.")
        self.assertTrue(len(fvt.feature_vectors) > 0)
        self.assertTrue(len(fvt.targets_for_features) > 0)

        # The 'search' and 'related' components together should be equal to the word2vec
        # portion of the full feature array.
        combined = list(fvt.feature_vectors_as_components[0]['search']) + \
                   list(fvt.feature_vectors_as_components[0]['related'])
        combined_array = np.array(combined)
        len_sentiment = len(fvt.feature_vectors_as_components[0]['sentiment'])
        just_w2v = fvt.feature_vectors[0][len_sentiment:]
        self.assertTrue(np.array_equal(combined_array, just_w2v))

    def test_empty_sentences(self):
        """
        Correctly handle empty sets of sentences.

        :return:
        """
        # Load the Configuration
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        # search_terms = ['strict', 'gun', 'control']
        search_terms = ['abortion', 'pro-life', 'pro-choice']

        # Token Filter
        tf = TokenFilter()

        # Initialize the Vector Model
        vector_model = SBertFeatureGenerator(False, False)

        # Process the Corpus
        topic = 'abortion'
        # rc = vdd.RawCorpusFourForums('gun control', database, host, user, password)
        rc = RawCorpusFourForums(topic, database, host, user, password)
        # rc.stance_a = 'prefers strict gun control'
        # rc.stance_b = 'opposes strict gun control'
        rc.stance_a = 'pro-life'
        rc.stance_b = 'pro-choice'
        corpus_stats = rc.print_stats()
        pdo = ParsedDocumentsFourForums(tf, topic, rc.stance_a,
                                        rc.stance_b, database, host, user, password)
        query_limit = 750
        print(f"Only pulling the first {query_limit} documents from the database")
        pdo.set_result_limit(query_limit)
        pdo.stance_agreement_cutoff = rc.stance_agreement_cutoff
        label_a = pdo.get_stance_label(rc.stance_a)
        label_b = pdo.get_stance_label(rc.stance_b)
        print(f"{rc.stance_a} => {label_a}, {rc.stance_b} => {label_b}")
        pdo.process_corpus()

        print("Finding Related Terms")
        fk = FindCharacteristicKeywords(pdo, print_stats=False)
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')
        print(f"{len(search_terms)} search terms and {len(related_terms)} related terms found for corpus.")

        context_size = 6
        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        print("Starting Feature Creation")
        start = time.process_time()
        fvt.create_feature_vectors_and_targets()
        end = time.process_time()
        print(f"Created {len(fvt.feature_vectors)} feature vectors in {(end - start) / 60:.2f} minutes.")

    def test_printing_embedding_sentences(self):
        """
        Can we print out the sentences that were used to create the embeddings

        :return:
        """
        """
        This is basically our end-to-end test case. Take some data from the database and create feature vectors
        from those documents. The test uses a smaller word2vec model for speed.
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(10)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        w2v_obj = SBertFeatureGenerator(False, False)
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, w2v_obj, search_terms, related_terms, context_size)
        fvt.create_feature_vectors_and_targets()
        print("First Feature Vector")
        print(fvt.feature_vectors[0])
        print("First Target Class")
        print(fvt.targets_for_features[0])
        print(f"Created {len(fvt.feature_vectors)} feature vectors.")
        self.assertTrue(len(fvt.feature_vectors) > 0)
        self.assertTrue(len(fvt.targets_for_features) > 0)

        contexts = fvt.contexts
        first_search_contexts = contexts['search'][0]
        print("\n## Printing Sentences Corresponding to Search Contexts from the First Document\n")
        for context in first_search_contexts:
            print(f"Matching Stem: {context.term}")
            print(f"Matched {len(context.contexts)} contexts")
            for s_i in context.sentence_indices:
                print(s_i)
                for i, sent in enumerate(pdo.all_docs[0].sents):
                    if i in s_i:
                        print(f"{i}: {sent.text}")

        first_related_contexts = contexts['related'][0]
        print("\n## Printing Sentences Corresponding to Related Contexts from the First Document\n")
        for context in first_related_contexts:
            print(f"Matching Stem: {context.term}")
            print(f"Matched {len(context.contexts)} contexts")
            for s_i in context.sentence_indices:
                print(s_i)
                for i, sent in enumerate(pdo.all_docs[0].sents):
                    if i in s_i:
                        print(f"{i}: {sent.text}")


if __name__ == '__main__':
    unittest.main()