import math
import re
import unittest
import configparser

from nltk import SnowballStemmer

from src.viewpointdiversitydetection.compare_documents import compute_avg_wmd_distance, ContextAndEntities
from src.viewpointdiversitydetection.compare_documents import create_document_index_combinations
from src.viewpointdiversitydetection.compare_documents import get_number_of_negations_from_span
from src.viewpointdiversitydetection.compare_documents import compute_avg_negation_difference
from src.viewpointdiversitydetection.compare_documents import GenerateSentencesAndContexts
from viewpointdiversitydetection import ParsedDocumentsFourForums
from viewpointdiversitydetection import TokenFilter
import gensim.downloader as api

# from negspacy.negation import Negex
from negspacy.termsets import termset
import spacy
# import torch


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
        wmds = [compute_avg_wmd_distance(contexts_by_docid[i[0]].contexts_and_entities,
                                         contexts_by_docid[i[1]].contexts_and_entities,
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

    def test_document_sentence_iterators(self):
        """
        A silly test class. As part of ensuring that we can pickle a SentencesAndContexts object we save our extracted
        sentences as Spacy Docs using the span.as_doc() method. This little test class is me making sure I understand
        how the doc.sents generator works.
        Every time you call doc.sents you get a new generator. So next(doc.sents) will always give you the first
        sentence in the doc object because you are creating a brand new generator. So, counter intuitively, calling
        next(doc.sents); next(doc.sents) returns the first sentence twice, not the first sentence and the second
        sentence.
        """
        ts = termset("en")
        nlp = spacy.load('en_core_web_trf')
        nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})

        doc1_text = "It is not Los Angeles."
        doc1 = nlp(doc1_text)
        doc2_sentence_1 = "This is my first sentence."
        doc2_sentence_2 = "Another sentence follows the first."
        doc2_text = doc2_sentence_1 + " " + doc2_sentence_2
        doc2 = nlp(doc2_text)

        # Test my assumptions for a single sentence document

        sentences = [s for s in doc1.sents]

        self.assertTrue(next(doc1.sents).text == doc1_text)
        self.assertTrue(next(doc1.sents).text == doc1_text)
        self.assertTrue(len(sentences) == 1)

        # This would not work with a normal iterator. The Spacy document object must do some magic to
        # rewind once you hit the end.
        self.assertTrue(sentences[0] == next(doc1.sents))

        # length of sentence in doc1
        self.assertTrue(len(next(doc1.sents)) == 6)
        # first token in the first sentence of doc 1
        self.assertTrue(next(doc1.sents)[0].text == 'It')
        # the second token in the first sentence of the doc
        self.assertTrue(next(doc1.sents)[1].text == 'is')

        # Test my assumptions about a two sentence document
        sentences = [s for s in doc2.sents]
        self.assertTrue(next(doc2.sents).text == doc2_sentence_1)
        self.assertTrue(next(doc2.sents).text == doc2_sentence_1)

        sent2_itr = doc2.sents
        self.assertTrue(next(sent2_itr).text == doc2_sentence_1)
        self.assertTrue(next(sent2_itr).text == doc2_sentence_2)
        with self.assertRaises(StopIteration):
            next(sent2_itr)

    def test_calculate_number_of_negations(self):
        """
        Test the calculation of the number of negations in a context. This test ensures that the ContextAndEntities
        object and extracting and correctly tracking the entities and their negations apart from the Spacy Span
        object.
        """
        # torch.set_num_threads(1)
        ts = termset("en")
        nlp = spacy.load('en_core_web_trf')
        nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})

        def create_context_and_entity_object(pdo, context_start, context_end):
            # convert the pdo and context boundaries into a ContextAndEntities object
            sentence_span = next(pdo.sents)
            context_tokens = [t for t in sentence_span]
            entities = [str(e) for e in sentence_span.ents]
            entities_start = [e.start for e in sentence_span.ents]
            entities_end = [e.end for e in sentence_span.ents]
            entities_are_negated = [e._.negex for e in sentence_span.ents]
            cae = ContextAndEntities(sentence_span, context_tokens, context_start, context_end, entities,
                                     entities_start, entities_end, entities_are_negated)
            return cae

        doc1 = nlp("It is not Los Angeles.")
        doc2 = nlp("It is Los Angeles.")
        doc3 = nlp("It is not Los Angeles and it is not Chicago")

        doc1_negations = create_context_and_entity_object(doc1, 0, 5).get_number_of_negations()
        self.assertTrue(doc1_negations == 1)
        self.assertTrue(doc1_negations == get_number_of_negations_from_span(next(doc1.sents), 0, 5))

        doc2_negations = create_context_and_entity_object(doc2, 0, 4).get_number_of_negations()
        self.assertTrue(doc2_negations == 0)
        self.assertTrue(doc2_negations == get_number_of_negations_from_span(next(doc2.sents), 0, 4))

        doc3_negations = create_context_and_entity_object(doc3, 0, 10).get_number_of_negations()
        self.assertTrue(doc3_negations == 2)
        self.assertTrue(doc3_negations == get_number_of_negations_from_span(next(doc3.sents), 0, 10))

        # There is only one negation within this context
        doc3_c2_negations = create_context_and_entity_object(doc3, 0, 5).get_number_of_negations()
        self.assertTrue(doc3_c2_negations == 1)
        self.assertTrue(doc3_c2_negations == get_number_of_negations_from_span(next(doc3.sents), 0, 5))

        # There are no negations within this context
        doc3_c3_negations = create_context_and_entity_object(doc3, 0, 2).get_number_of_negations()
        self.assertTrue(doc3_c3_negations == 0)
        self.assertTrue(doc3_c3_negations == get_number_of_negations_from_span(next(doc3.sents), 0, 2))

        # If the negated entity overlaps with the boundaries of the context
        # we still count it as a negation in the context
        doc3_c4_negations = create_context_and_entity_object(doc3, 0, 4).get_number_of_negations()
        self.assertTrue(doc3_c4_negations == 1)
        self.assertTrue(doc3_c4_negations == get_number_of_negations_from_span(next(doc3.sents), 0, 4))

    def test_original_index_logic(self):
        """
        Ensure that the logic for extracting the context and, more difficult, the original beginning and ending
        of that context in terms of the original unfiltered token list, is correct.
        """
        doc_text = """
I read the miller decision last night (after searching futily for half an hour for the Warren 
decision), and I have to say that it was perhaps the most pro-gun piece of literature I have ever read. 
"THE COURT CAN NOT TAKE JUDICIAL NOTICE THAT A SHOTGUN HAVING A BARREL
LESS THAN 18 INCHES LONG HAS TODAY ANY REASONABLE RELATION TO THE
PRESERVATION OR EFFICIENCY OF A WELL REGULATED MILITIA; AND THEREFORE
CAN NOT SAY THAT THE SECOND AMENDMENT GUARANTEES TO THE CITIZEN THE
RIGHT TO KEEP AND BEAR SUCH A WEAPON. """
        ts = termset("en")
        nlp = spacy.load('en_core_web_trf')
        nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})
        doc = nlp(doc_text)
        stemmer = SnowballStemmer(language='english')

        # create a dummy SentencesAndContexts object, we only need it so that we can
        # test the _get_context_tokens method.
        context_generator = GenerateSentencesAndContexts('gun', 6, stemmer)
        s_and_c_obj = context_generator.generate(doc)

        tf = context_generator.token_filter
        sentences = [s for s in doc.sents]
        #
        # First Sentence
        #

        s = sentences[0]
        # convert the sentence <span> into tokens, keeping track of each tokens original index
        unfiltered_tokens = [str(t) for t in s]
        raw_tokens_and_indexes = [(i, str(t)) for i, t in enumerate(s) if tf.filter(t)]
        raw_tokens = [i[1] for i in raw_tokens_and_indexes]
        original_indices = [i[0] for i in raw_tokens_and_indexes]
        stemmed_tokens = [stemmer.stem(t) for t in raw_tokens]  # convert the sentence <span> into tokens
        if 'gun' in stemmed_tokens:  # see if we have a matching term
            term_index = stemmed_tokens.index('gun')  # note getting matching term using the
            original_index = original_indices[term_index]
            self.assertTrue(original_index == 36)
            self.assertTrue(term_index == 14)
            self.assertTrue(original_indices[14] == 36)
            term_start = term_index - 6
            if term_start < 0:
                term_start = 0
            original_start = original_indices[term_start]
            term_end = term_index + 6 + 1
            if term_end > len(stemmed_tokens):
                term_end = len(stemmed_tokens)
            original_end = original_indices[term_end - 1] + 1

            context, context_start, context_end = s_and_c_obj._get_context_tokens(unfiltered_tokens, raw_tokens,
                                                                                  term_index, original_indices)
            self.assertTrue(context == unfiltered_tokens[original_start:original_end])
            self.assertTrue(context_start == original_start)
            self.assertTrue(context_end == original_end)

        #
        # Second Sentence
        #

        s = sentences[2]
        # convert the sentence <span> into tokens, keeping track of each tokens original index
        unfiltered_tokens = [str(t) for t in s]
        raw_tokens_and_indexes = [(i, str(t)) for i, t in enumerate(s) if tf.filter(t)]
        raw_tokens = [i[1] for i in raw_tokens_and_indexes]
        original_indices = [i[0] for i in raw_tokens_and_indexes]
        stemmed_tokens = [stemmer.stem(t) for t in raw_tokens]  # convert the sentence <span> into tokens
        if 'shotgun' in stemmed_tokens:  # see if we have a matching term
            term_index = stemmed_tokens.index('shotgun')  # note getting matching term using the
            original_index = original_indices[term_index]
            self.assertTrue(original_index == 10)
            self.assertTrue(term_index == 5)
            self.assertTrue(original_indices[5] == 10)
            term_start = term_index - 6
            if term_start < 0:
                term_start = 0
            original_start = original_indices[term_start]
            term_end = term_index + 6 + 1
            if term_end > len(stemmed_tokens):
                term_end = len(stemmed_tokens)
            original_end = original_indices[term_end - 1] + 1

            context, context_start, context_end = s_and_c_obj._get_context_tokens(unfiltered_tokens, raw_tokens,
                                                                                  term_index, original_indices)
            self.assertTrue(context == unfiltered_tokens[original_start:original_end])
            self.assertTrue(context_start == original_start)
            self.assertTrue(context_end == original_end)

    def test_total_number_of_negations(self):
        """
        Test the total number of negations in a slice of the corpus.
        """
        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 50
        pdo.set_result_limit(limit)
        pdo.stance_agreement_cutoff = 1  # disable the inter-annotator cutoff
        pdo.extract_negations = True
        pdo.spacy_model = 'en_core_web_trf'
        pdo.process_corpus()
        stemmer = SnowballStemmer(language='english')

        context_generator = GenerateSentencesAndContexts('shotgun', 10, stemmer)
        contexts_and_spans_by_docid = [context_generator.generate(doc, return_spans=True) for doc in pdo.all_docs]

        negations_doc = list()
        negations_span = list()
        for spans, sentences_and_contexts_obj in contexts_and_spans_by_docid:
            for span, cae in zip(spans, sentences_and_contexts_obj.contexts_and_entities):
                start = cae.context_start
                end = cae.context_end
                negations_doc.append(cae.get_number_of_negations())
                negations_span.append(get_number_of_negations_from_span(span, start, end))
        print("Docs")
        print(negations_doc)
        print(sum(negations_doc))
        print("Spans")
        print(negations_span)
        print(sum(negations_span))
        # We have a few negated entities in this set of documents
        self.assertTrue(sum(negations_doc) > 0)
        # The number of negated entities counted directly from the span should be the same as the
        # number of negations counted from the ContextsAndEntities object.
        self.assertTrue(negations_doc == negations_span)

    def test_average_negation_difference(self):
        """
        Test our computation of the average negation difference between documents
        """
        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 50
        pdo.set_result_limit(limit)
        pdo.stance_agreement_cutoff = 1  # disable the inter-annotator cutoff
        pdo.extract_negations = True
        pdo.spacy_model = 'en_core_web_trf'
        pdo.process_corpus()
        stemmer = SnowballStemmer(language='english')

        context_generator = GenerateSentencesAndContexts('shotgun', 6, stemmer)
        contexts_by_docid = [context_generator.generate(doc) for doc in pdo.all_docs]

        document_index_combinations = create_document_index_combinations(pdo)

        negation_difference = [compute_avg_negation_difference(contexts_by_docid[i[0]], contexts_by_docid[i[1]])
                               for i in document_index_combinations]
        print(negation_difference)

        # print(negation_difference)
        self.assertTrue(sum(negation_difference) > 0.0)  # we should find at least 1 thing ...


if __name__ == '__main__':
    unittest.main()
