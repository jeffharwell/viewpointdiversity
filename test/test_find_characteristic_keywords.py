import unittest
import configparser
from nltk.corpus import stopwords
from src.viewpointdiversitydetection import ParsedDocumentsFourForums
from src.viewpointdiversitydetection import FindCharacteristicKeywords
from src.viewpointdiversitydetection import TokenFilter


class FindCharacteristicKeywordsTest(unittest.TestCase):
    """
    A few tests for the FindCharacteristicKeywords class. These some of these assume that you have a connection
    to a copy of the Internet Argument Corpus database. Put the database host, username, and password in the
    config.ini file before testing.
    """

    def test_context_extraction_no_content(self):
        """
        Make sure that a stance is still invalid even if it is the substring of a valid stance.
        """

        config = configparser.ConfigParser()
        config.read('config.ini')

        tf = TokenFilter()

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control', database, host, user, password)

        with self.assertRaises(ValueError):
            FindCharacteristicKeywords(pdo)

    def test_context_extraction_guns(self):
        """
        Test the Context Extraction Method
        """

        config = configparser.ConfigParser()
        config.read('config.ini')

        tf = TokenFilter()

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(1000)
        pdo.process_corpus()
        terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        print("\n-- Extracted nouns related to the search terms")
        search_nouns = fk.get_unique_nouns_from_term_context(terms, 'search')
        self.assertTrue(len(search_nouns) > 0)
        print("\n-- Extracted nouns related to the related terms")
        related_nouns = fk.get_unique_nouns_from_term_context(search_nouns, 'related')
        self.assertTrue(len(related_nouns) > 0)
        print("")
        print(f"Search Nouns {len(search_nouns)}: {search_nouns}")
        print(f"Related Nouns: {len(related_nouns)}: {related_nouns}")
