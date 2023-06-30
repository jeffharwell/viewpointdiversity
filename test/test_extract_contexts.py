import re
import unittest
import configparser
from src.viewpointdiversitydetection import ParsedDocumentsCommonCrawl
from src.viewpointdiversitydetection import ExtractContexts
from viewpointdiversitydetection import TokenFilter
import json
import decimal
import pandas


class ParsedDocumentCommonCrawlTest(unittest.TestCase):
    """
    A few tests for the ParsedDocumentsFourForums class. These some of these assume that you have a connection
    to a copy of the Internet Argument Corpus database. Put the database host, username, and password in the
    config.ini file before testing.
    """
    def setUp(self):
        # Load our sample data
        with open('./resources/cc_sample_df.json') as f:
            cc_sample_dict = json.load(f)
            for r in cc_sample_dict:
                # Had to make our decimals a string for the JSON file, convert
                # them back to decimal objects here
                r['relevance'] = decimal.Decimal(r['relevance'])
            cc_sample_df = pandas.DataFrame(cc_sample_dict)

        self.cc_df = cc_sample_df
        self.tf = TokenFilter()

    def test_extracts_contexts(self):
        """
        Test one-shot context extraction
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus()

        # Get the number of documents from our test data set
        num_docs = len(self.cc_df)

        # After the parse each of the relevant datastructures
        assert(len(cc_pdo.all_docs) == num_docs)
        assert(len(cc_pdo.warc_record_id) == num_docs)
        assert(len(cc_pdo.relevance) == num_docs)

        # Extract some contexts
        ec = ExtractContexts(cc_pdo, 6)
        ec.extract_contexts({'search': ['abortion'], 'related': ['morality']})

        # We should record the search terms
        assert(len(ec.terms_to_extract['search']) == 1)
        assert(len(ec.terms_to_extract['related']) == 1)
        # extract search contexts from each of the documents
        assert(len(ec.contexts['search']) == num_docs)
        # and extracted related contexts from only 1
        assert(len(ec.contexts['related']) == 1)

    def test_extract_additional_contexts(self):
        """
        Test incremental context extraction via the extract_additional_contexts method
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus()

        # Get the number of documents from our test data set
        num_docs = len(self.cc_df)

        # After the parse each of the relevant datastructures
        assert(len(cc_pdo.all_docs) == num_docs)
        assert(len(cc_pdo.warc_record_id) == num_docs)
        assert(len(cc_pdo.relevance) == num_docs)

        # Extract some contexts
        ec = ExtractContexts(cc_pdo, 6)
        ec.extract_contexts({'search': ['abortion'], 'related': ['morality']})

        # We should record the search terms
        assert(len(ec.terms_to_extract['search']) == 1)
        assert(len(ec.terms_to_extract['related']) == 1)
        # extract search contexts from each of the documents
        assert(len(ec.contexts['search']) == num_docs)
        # and extracted related contexts from only 1
        assert(len(ec.contexts['related']) == 1)

        ec.extract_additional_contexts({'related': ['anti-holocaust', 'Romney']})
        print(ec.terms_to_extract['related'])
        assert(len(ec.terms_to_extract['related']) == 3)
        assert(len(ec.contexts['related']) == 2)  # we should add a context
        assert (len(ec.contexts['search']) == num_docs)  # or mess with the results of other contexts

        # doing it again shouldn't duplicate anything
        ec.extract_additional_contexts({'related': ['anti-holocaust', 'Romney']})
        assert(len(ec.terms_to_extract['related']) == 3)
        assert(len(ec.contexts['related']) == 2)  # we should not add any new documents
        assert(len(ec.contexts['search']) == num_docs)  # or mess with the results of other contexts

        # doing it again shouldn't duplicate anything
        ec.extract_additional_contexts({'search': ['abortion'], 'related': ['morality']})
        assert(len(ec.terms_to_extract['related']) == 3)
        assert(len(ec.contexts['related']) == 2)  # we should not add any new documents
        assert(len(ec.contexts['search']) == num_docs)  # or mess with the results of other contexts

        # adding to the search term shouldn't increase the number of document matches
        # but it should increase the total number of TermContexts objects we have fished out
        total_contexts = 0
        for doc_id, tcs in ec.contexts['search'].items():
            # Count the number of total extracted contexts per term
            # note that each TermContext can have multiple contexts for this single term
            # stored in the .contexts object variable
            total_contexts += len(tcs)
        ec.extract_additional_contexts({'search': ['abortion', 'evangelical', 'vegetarian'], 'related': ['morality']})
        assert(len(ec.terms_to_extract['related']) == 3)  # related should stay the same
        assert(len(ec.terms_to_extract['search']) == 3)  # search is now three distinct keywords
        assert(len(ec.contexts['related']) == 2)  # we should not add any new documents
        assert(len(ec.contexts['search']) == num_docs)  # or mess with the results of other contexts
        new_total_contexts = 0
        for doc_id, tcs in ec.contexts['search'].items():
            # Count the number of total extracted contexts per term
            # note that each TermContext can have multiple contexts for this single term
            # stored in the .contexts object variable
            new_total_contexts += len(tcs)
        # We should have more contexts extracted per term, although the number of documents
        # is the same.
        assert(new_total_contexts > total_contexts)

    def test_automatically_perform_initial_extraction_first(self):
        """
        Test incremental context extraction via the extract_additional_contexts method
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus()

        # Get the number of documents from our test data set
        num_docs = len(self.cc_df)

        # After the parse each of the relevant datastructures
        assert(len(cc_pdo.all_docs) == num_docs)
        assert(len(cc_pdo.warc_record_id) == num_docs)
        assert(len(cc_pdo.relevance) == num_docs)

        # Extract some contexts
        ec = ExtractContexts(cc_pdo, 6)
        ec.extract_additional_contexts({'search': ['abortion'], 'related': ['morality']})

        # We should record the search terms
        assert(len(ec.terms_to_extract['search']) == 1)
        assert(len(ec.terms_to_extract['related']) == 1)
        # extract search contexts from each of the documents
        assert(len(ec.contexts['search']) == num_docs)
        # and extracted related contexts from only 1
        assert(len(ec.contexts['related']) == 1)


if __name__ == '__main__':
    unittest.main()
