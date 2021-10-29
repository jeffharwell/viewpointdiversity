import unittest
import spacy
import torch

from src.viewpointdiversitydetection import NounTokenFilter


class TokenFilterTest(unittest.TestCase):
    """
    Test the TokenFilter class.
    """

    def setUp(self):
        """
        Set up our tokens.
        """
        text = "This is my test text and  sentence."
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
        self.invalid_token = mydoc[0]
        self.punct_token = mydoc[8]

    def test_stop_word(self):
        """
        This is basically our end-to-end test case. Take some data from the database and create feature vectors
        from those documents. The test uses a smaller word2vec model for speed.
        """

        tf = NounTokenFilter()

        self.assertTrue(tf.is_stop_word(self.stopword_token))

    def test_filter_stopwords(self):
        """
        Does the filter remove a stop word.
        """
        tf = NounTokenFilter()
        self.assertFalse(tf.filter(self.stopword_token))

    def test_filter_space(self):
        """
        Does the filter remove space token.
        """
        tf = NounTokenFilter()
        self.assertFalse(tf.filter(self.space_token))

    def test_filter_punctuation(self):
        """
        Does the filter remove punctuation token.
        """
        tf = NounTokenFilter()
        self.assertFalse(tf.filter(self.punct_token))

    def test_pass_valid_token(self):
        """
        Does the filter pass a non space, punctuation, or stop word token?
        """
        tf = NounTokenFilter()
        self.assertTrue(tf.filter(self.valid_token))

    def test_reject_invalid_token(self):
        """
        Does the filter reject a non-noun token?
        """
        tf = NounTokenFilter()
        self.assertFalse(tf.filter(self.invalid_token))


if __name__ == '__main__':
    unittest.main()
