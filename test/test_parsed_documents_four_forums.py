import re
import unittest
import configparser
from src.viewpointdiversitydetection import ParsedDocumentsFourForums
from viewpointdiversitydetection import TokenFilter


class ParsedDocumentFourForumTest(unittest.TestCase):
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

    def test_invalid_stance_as_substring(self):
        """
        Make sure that a stance is still invalid even if it is the substring of a valid stance.
        """

        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(self.tf, 'gun control', 'oppose strict gun control',
                                      'prefers strict gun control', self.database, self.host, self.user, self.password)

    def test_invalid_topic(self):
        """
        Test class to ensure that the function create_has_sentiments_present_vector is actually working.
        """

        # an invalid topic, it should throw a value error
        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(self.tf, 'invalid_topic', 'stance_a', 'stance_b',
                                      self.database, self.host, self.user, self.password)

        # an invalid stance, should also throw a value error
        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(self.tf, 'climate change', 'stance_a', 'humans not responsible',
                                      self.database, self.host, self.user, self.password)
        # an invalid stance, should also throw a value error
        with self.assertRaises(ValueError):
            ParsedDocumentsFourForums(self.tf, 'climate change', 'humans responsible', 'stance_b',
                                      self.database, self.host, self.user, self.password)

        # valid topic and stances, initialize without an error
        ParsedDocumentsFourForums(self.tf, 'climate change', 'humans responsible', 'humans not responsible',
                                  self.database, self.host, self.user, self.password)

    def test_print_annotated_posts_by_topic(self):
        """
        Not much of a test class. Just does a bit of checking to see if this method actually
        returns some data.
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'climate change', 'humans responsible', 'humans not responsible',
                                        self.database, self.host, self.user, self.password)
        stats = pdo.print_annotated_posts_by_topic()
        # print(stats)

        self.assertGreater(stats['gun control'], 100)

    def test_cant_set_limit_twice(self):
        """
        Because of the way we are doing query construction within the class, you can't set change query limit
        once you set it. Ensure that attempting to do this throws a RuntimeException.
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        pdo.set_result_limit(100)
        with self.assertRaises(RuntimeError):
            pdo.set_result_limit(200)

    def test_limit_query_results(self):
        """
        Test the ability of the ParsedDocumentsFourForumns class to limit the number of texts used when parsing.
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
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

    def test_get_stance_labels(self):
        """
        Test the ability of the ParsedDocumentsFourForums class to limit the number of texts used when parsing.
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)

        with self.assertRaises(ValueError):
            pdo.get_stance_label('invalid stance name')

        self.assertTrue(pdo.get_stance_label('opposes strict gun control') == 'a')
        self.assertTrue(pdo.get_stance_label('prefers strict gun control') == 'b')


if __name__ == '__main__':
    unittest.main()
