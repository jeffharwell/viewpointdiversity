import unittest
from src.viewpointdiversitydetection import ParsedDocumentsCommonCrawl
from viewpointdiversitydetection import TokenFilter
from viewpointdiversitydetection import TopicFeatureGenerator
import json
import decimal
import pandas


class TopicFeatureGeneratorTest(unittest.TestCase):
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

    def test_topic_vector_generation(self):
        """
        Test to see if the class successfully loads a data frame
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus(print_stats=False)

        # Get the number of documents from our test data set
        num_docs = len(self.cc_df)

        # Create Topic Vectors from training
        topic_vector_obj = TopicFeatureGenerator()
        topic_vector_obj.workers = 7  # good for an 8 core CPU
        topic_vector_obj.debug = False  # we want the debug output
        topic_vector_obj.min_number_topics = 5  # max coherence seems to land at 4 .. which is a bit low
        topic_vector_obj.create_topic_vectors_from_texts(cc_pdo.text)
        print(f"Created a topic vector with {topic_vector_obj.num_topics} topics from {len(cc_pdo.text)} texts.")
        print(f"Coherence of the LDA model is {topic_vector_obj.coherence_score}")
        train_topic_vectors = topic_vector_obj.topic_vectors

        print(train_topic_vectors)
        print(f"Generated {len(train_topic_vectors)}")

        # After the parse each of the relevant datastructures
        assert(len(cc_pdo.all_docs) == len(train_topic_vectors))
        assert(len(train_topic_vectors[0]) == topic_vector_obj.num_topics)

    def test_topic_vector_generation_new_text(self):
        """
        Test to see if the class successfully loads a data frame
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus(print_stats=False)

        train_text = cc_pdo.text[:3]
        new_text = cc_pdo.text[3:5]

        # Get the number of documents from our test data set
        num_docs = len(train_text)

        # Create Topic Vectors from training
        topic_vector_obj = TopicFeatureGenerator()
        topic_vector_obj.workers = 7  # good for an 8 core CPU
        topic_vector_obj.debug = False  # we want the debug output
        topic_vector_obj.min_number_topics = 5  # max coherence seems to land at 4 .. which is a bit low
        topic_vector_obj.create_topic_vectors_from_texts(train_text)
        print(f"Created a topic vector with {topic_vector_obj.num_topics} topics from {len(train_text)} texts.")
        print(f"Coherence of the LDA model is {topic_vector_obj.coherence_score}")

        # Create the topic vectors for the new text
        test_topic_vectors = topic_vector_obj.create_topic_vectors_from_new_text(new_text)
        print("Topic Vectors from New Text:")
        print(test_topic_vectors)

        assert(len(new_text) == len(test_topic_vectors))
        assert(len(test_topic_vectors[0]) == topic_vector_obj.num_topics)


if __name__ == '__main__':
    unittest.main()
