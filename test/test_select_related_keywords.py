import re
import unittest
import configparser

from nltk import SnowballStemmer

from src.viewpointdiversitydetection import ParsedDocumentsCommonCrawl
from src.viewpointdiversitydetection import ExtractContexts
from viewpointdiversitydetection import TokenFilter
from viewpointdiversitydetection import SelectRelatedKeywords
import json
import decimal
import pandas


class SelectRelatedKeywordsTest(unittest.TestCase):
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
        self.stemmer = SnowballStemmer(language='english')

    def test_value_error_if_no_contexts_extracted(self):
        """
        A simple test of class functionality
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus(print_stats=False)

        k = 100
        # First extract our related terms given the value of k
        search_terms = ['prochoice']

        def call_constructor():
            SelectRelatedKeywords(cc_pdo, list(range(0, len(cc_pdo.all_docs))),
                                  search_terms, self.stemmer)

        self.assertRaises(ValueError, call_constructor)

    def test_general_extraction(self):
        """
        A simple test of class functionality
        :return:
        """
        context_size = 6
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus(print_stats=False)

        k = 100
        # First extract our related terms given the value of k
        search_terms = ['abortion']
        srk = SelectRelatedKeywords(cc_pdo, list(range(0, len(cc_pdo.all_docs))), search_terms, self.stemmer)
        related_keywords = srk.get_related_keywords_context_threshold(k, context_size)

        # We should have extracted some related keywords
        assert(len(related_keywords) > 0)
        # And doctor should be among them
        assert('doctors' in related_keywords)


if __name__ == '__main__':
    unittest.main()
