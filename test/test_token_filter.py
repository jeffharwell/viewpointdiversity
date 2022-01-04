import unittest
import spacy
import torch

from src.viewpointdiversitydetection import TokenFilter


class TokenFilterTest(unittest.TestCase):
    """
    Test the TokenFilter class.
    """

    def setUp(self):
        """
        Set up our tokens.
        """
        text = "This is my test text and  sentence http://cn https://k."
        texts = [text]
        # Set up Spacy
        torch.set_num_threads(1)  # works around a Spacy bug
        nlp = spacy.load('en_core_web_sm')
        doc_list = nlp.pipe(texts, n_process=4)
        all_docs = []
        for doc in doc_list:
            all_docs.append(doc)

        mydoc = all_docs[0]
        """
        for i, token in enumerate(mydoc):
            print(f"Index: {i}, Token: {token}")
            print(token.is_space)
            print(token.is_punct)
            print(tf.is_stop_word(token))
        print(mydoc)
        """

        self.space_token = mydoc[6]
        self.stopword_token = mydoc[5]
        self.valid_token = mydoc[7]
        self.http_token = mydoc[8]
        self.https_token = mydoc[9]
        self.punct_token = mydoc[10]

    def test_filter_http(self):
        tf = TokenFilter()

        self.assertFalse(tf.filter(self.http_token))

    def test_filter_https(self):
        tf = TokenFilter()

        self.assertFalse(tf.filter(self.https_token))

    def test_stop_word(self):

        tf = TokenFilter()

        self.assertTrue(tf.is_stop_word(self.stopword_token))

    def test_filter_stopwords(self):
        """
        Does the filter remove a stop word.
        """
        tf = TokenFilter()
        self.assertFalse(tf.filter(self.stopword_token))

    def test_filter_space(self):
        """
        Does the filter remove space token.
        """
        tf = TokenFilter()
        self.assertFalse(tf.filter(self.space_token))

    def test_filter_punctuation(self):
        """
        Does the filter remove punctuation token.
        """
        tf = TokenFilter()
        self.assertFalse(tf.filter(self.punct_token))

    def test_pass_valid_token(self):
        """
        Does the filter pass a non space, punctuation, or stop word token?
        """
        tf = TokenFilter()
        self.assertTrue(tf.filter(self.valid_token))


if __name__ == '__main__':
    unittest.main()
