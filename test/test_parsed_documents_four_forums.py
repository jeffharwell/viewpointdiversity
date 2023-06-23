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

    def test_pre_transform(self):
        """

        :return:
        """
        def get_transform():
            prolife_pattern_lower = re.compile(r'[p][Rr][Oo][\s-]*[Ll][Ii][Ff][Ee]')
            prochoice_pattern_lower = re.compile(r'[p][Rr][Oo][\s-]*[Cc][Hh][Oo][Ii][Cc][Ee]')
            prolife_pattern_upper = re.compile(r'[P][Rr][Oo][\s-]*[Ll][Ii][Ff][Ee]')
            prochoice_pattern_upper = re.compile(r'[P][Rr][Oo][\s-]*[Cc][Hh][Oo][Ii][Cc][Ee]')

            def transform_text(text):
                """
                t1 = re.sub(prolife_pattern_lower, 'prolife', text)
                t2 = re.sub(prochoice_pattern_lower, 'prochoice', t1)
                t3 = re.sub(prolife_pattern_upper, 'Prolife', t2)
                t4 = re.sub(prochoice_pattern_upper, 'Prochoice', t3)
                """
                t1 = prolife_pattern_lower.sub('prolife', text)
                t2 = prochoice_pattern_lower.sub('prochoice', t1)
                t3 = prolife_pattern_upper.sub('Prolife', t2)
                t4 = prochoice_pattern_upper.sub('Prochoice', t3)
                return t4

            return transform_text

        topic = 'abortion'
        pdo = ParsedDocumentsFourForums(self.tf, topic, 'pro-life',
                                        'pro-choice', self.database, self.host, self.user, self.password)
        pdo.set_pre_transform(get_transform())
        pdo.set_result_limit(2000)
        pdo.process_corpus()
        assert('prolife' in pdo.text[39])
        assert('pro-life' in pdo.raw_text[39])

        assert('Pro-life' in pdo.raw_text[391])
        assert('Prolife' in pdo.text[391])

        assert('pro life' in pdo.raw_text[27])
        assert('prolife' in pdo.text[27])

        assert('Pro-choice' in pdo.raw_text[163])
        assert('Prochoice' in pdo.text[163])

        assert('pro-choice' in pdo.raw_text[65])
        assert('prochoice' in pdo.text[65])

        assert('pro choice' in pdo.raw_text[1063])
        assert('prochoice' in pdo.text[1063])

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

    def test_add_negation(self):
        """
        Tests the option to add Negex to the Spacy pipeline by setting an attribute
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 40
        pdo.set_result_limit(limit)
        pdo.extract_negations = True
        pdo.spacy_model = 'en_core_web_trf'
        print("Testing Add Negation")
        pdo.process_corpus()

        print("Listing All Named Entities and Negation Status")
        for d in pdo.all_docs:
            for e in d.ents:
                print(e.text, e._.negex)

    def test_negation_and_tokenize(self):
        """
        Should throw a runtime error if you try to extract negations but only tokenize the documents.
        """
        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 40
        pdo.set_result_limit(limit)
        pdo.extract_negations = True
        pdo.tokenize_only = True
        pdo.spacy_model = 'en_core_web_trf'
        with self.assertRaises(RuntimeError):
            pdo.process_corpus()

    def test_tokenize_only(self):
        """
        If you set 'tokenize_only' to True it should disable all components in the pipeline and
        just tokenize the documents.
        """
        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 40
        pdo.set_result_limit(limit)
        pdo.tokenize_only = True
        pdo.spacy_model = 'en_core_web_trf'
        pdo.process_corpus()
        for d in pdo.all_docs:
            for t in d:
                self.assertEqual(t.pos_, '')
                self.assertIsInstance(t.is_space, bool)
                self.assertIsInstance(t.is_punct, bool)

    def test_continuous_target(self):
        """
        Tests the option to add Negex to the Spacy pipeline by setting an attribute
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 40
        pdo.set_result_limit(limit)
        pdo.stance_agreement_cutoff = 1  # disable the inter-annotator cutoff
        pdo.extract_negations = True
        pdo.spacy_model = 'en_core_web_trf'
        print("Testing Continuous Target")
        pdo.process_corpus()

        print(pdo.continuous_target)
        print(pdo.target)
        self.assertTrue(max(pdo.continuous_target) <= 1)
        self.assertTrue(min(pdo.continuous_target) >= -1)
        number_zeros = sum([1 for i in pdo.continuous_target if i == 0.0])

        # There should be no zeros in continuous target. That would mean that there were equal votes
        # for each stance. We filter these documents out because the binary target ends of being undefined.
        self.assertTrue(number_zeros == 0)

        # Need two more tests, one makes sure that everything above 0.0 is stance 'b', the other makes sure everything
        # below 0.0 is stance 'a'
        for c, t in zip(pdo.continuous_target, pdo.target):
            # print(c, t)
            if c > 0.0:
                self.assertTrue(t == 'b')
            if c < 0.0:
                self.assertTrue(t == 'a')

    def test_limit_query_results(self):
        """
        Test the ability of the ParsedDocumentsFourForumns class to limit the number of texts used when parsing.
        """

        pdo = ParsedDocumentsFourForums(self.tf, 'gun control', 'opposes strict gun control',
                                        'prefers strict gun control',
                                        self.database, self.host, self.user, self.password)
        limit = 100
        pdo.set_result_limit(limit)
        pdo.spacy_model = 'en_core_web_trf'
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
