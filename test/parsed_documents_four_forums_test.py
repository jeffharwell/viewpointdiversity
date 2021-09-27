import re
import unittest
import configparser
from src.viewpointdiversitydetection import ParsedDocumentsFourForums


class ParsedDocumentFourForumTest(unittest.TestCase):
    """
    A few tests for the ParsedDocumentsFourForums class. These some of these assume that you have a connection
    to a copy of the Internet Argument Corpus database. Put the database host, username, and password in the
    config.ini file before testing.
    """

    def test_invalid_stance_as_substring(self):
        """
        Make sure that a stance is still invalid even if it is the substring of a valid stance.
        """

        config = configparser.ConfigParser()
        config.read('config.ini')

        def dummy_filter(t):
            return True

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(dummy_filter, 'gun control', 'oppose strict gun control',
                                      'prefers strict gun control', database, host, user, password)

    def test_invalid_topic(self):
        """
        Test class to ensure that the function create_has_sentiments_present_vector is actually working.
        """

        # Read in our configuration from the config.ini file, it assumes we are
        # using the config in the src tree
        config = configparser.ConfigParser()
        config.read('config.ini')

        def dummy_filter(t):
            return True

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        # an invalid topic, it should throw a value error
        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(dummy_filter, 'invalid_topic', 'stance1', 'stance2',
                                      database, host, user, password)

        # an invalid stance, should also throw a value error
        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(dummy_filter, 'climate change', 'stance1', 'humans not responsible',
                                      database, host, user, password)
        # an invalid stance, should also throw a value error
        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(dummy_filter, 'climate change', 'humans responsible', 'stance2',
                                      database, host, user, password)

        # valid topic and stances, initialize without an error
        ParsedDocumentsFourForums(dummy_filter, 'climate change', 'humans responsible', 'humans not responsible',
                                  database, host, user, password)

    def test_print_annotated_posts_by_topic(self):
        """
        Not much of a test class. Just does a bit of checking to see if this method actually
        returns some data.
        """
        # Read in our configuration from the config.ini file, it assumes we are
        # using the config in the src tree
        config = configparser.ConfigParser()
        config.read('config.ini')

        def dummy_filter(t):
            return True

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(dummy_filter, 'climate change', 'humans responsible', 'humans not responsible',
                                        database, host, user, password)
        stats = pdo.print_annotated_posts_by_topic()
        # print(stats)

        self.assertGreater(stats['gun control'], 100)

    def test_print_raw_stats(self):
        """
        Not much of a test class. Just spits the raw corpus stats out.
        """
        # Read in our configuration from the config.ini file, it assumes we are
        # using the config in the src tree
        config = configparser.ConfigParser()
        config.read('config.ini')

        def dummy_filter(t):
            return True

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(dummy_filter, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control', database, host, user, password)
        stats = pdo.print_raw_corpus_stats()
        self.assertGreater(stats['discussion_topics_in_db'], 10)
        self.assertGreater(stats['posts_in_db'], 10)
        self.assertGreater(stats['authors_in_db'], 10)
        self.assertGreater(stats['posts_with_usable_stance'], 10)

    def test_cant_set_limit_twice(self):
        """
        Because of the way we are doing query construction within the class, you can't set change query limit
        once you set it. Ensure that attempting to do this throws a RuntimeException.
        """
        # Read in our configuration from the config.ini file, it assumes we are
        # using the config in the src tree
        config = configparser.ConfigParser()
        config.read('config.ini')

        def dummy_filter(t):
            return True

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(dummy_filter, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control', database, host, user, password)
        pdo.set_result_limit(100)
        with self.assertRaises(RuntimeError):
            pdo.set_result_limit(200)

    def test_limit_query_results(self):
        """
        Test the ability of the ParsedDocumentsFourForumns class to limit the number of texts used when parsing.
        """
        # Read in our configuration from the config.ini file, it assumes we are
        # using the config in the src tree
        config = configparser.ConfigParser()
        config.read('config.ini')

        def dummy_filter(t):
            return True

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(dummy_filter, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control', database, host, user, password)
        limit = 100
        pdo.set_result_limit(limit)
        self.assertTrue(pdo.is_limited)

        # Get the last line of the query, watch out for that ending newline which might be there
        query_list = pdo.query.split("\n")
        if len(query_list[-1]) > 0:
            last_line = query_list[-1]
        else:
            last_line = query_list[-2]

        pattern = r"\s*limit %s$" % limit
        self.assertTrue(re.match(pattern, last_line))

        pdo.process_corpus()
        self.assertTrue(len(pdo.text) <= limit)


if __name__ == '__main__':
    unittest.main()
