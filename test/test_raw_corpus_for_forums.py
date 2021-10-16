import unittest
import configparser
from src.viewpointdiversitydetection import RawCorpusFourForums


class RawCorpusFourForumTest(unittest.TestCase):

    def test_print_raw_stats(self):
        """
        Not much of a test class. Just spits the raw corpus stats out.
        """
        # Read in our configuration from the config.ini file, it assumes we are
        # using the config in the src tree
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        rc = RawCorpusFourForums('gun control', database, host, user, password)
        rc.stance_a = 'opposes strict gun control'
        rc.stance_b = 'prefers strict gun control'
        stats = rc.print_stats()
        self.assertGreater(stats['discussion_topics_in_db'], 10)
        self.assertGreater(stats['posts_in_db'], 10)
        self.assertGreater(stats['authors_in_db'], 10)
        self.assertGreater(stats['posts_with_usable_stance'], 10)

    def test_print_stances(self):
        """
        Do we properly print out the available stances for the topic
        """
        # Read in our configuration from the config.ini file, it assumes we are
        # using the config in the src tree
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        rc = RawCorpusFourForums('gun control', database, host, user, password)
        stances = rc.get_valid_stances()
        self.assertTrue('prefers strict gun control' in stances)
        self.assertTrue('opposes strict gun control' in stances)
