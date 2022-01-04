from nltk.corpus import stopwords

"""
TokenFilter module. Defines a basic TokenFilter class that can be used with the
ParsedDocumentsFourForums class and the NounTokenFilter used in the FindCharacteristicKeywords class.
"""


class TokenFilter:
    """
    A simple token filter. Can be used to control which words the ParsedDocumentsFourForums class considers
    valid. If you want to use a different set of stopwords then set the 'stop_words' attribute to your own
    list after initialisation. If you want to set up your own filter logic create a child class and override
    the 'filter' method.
    """

    def __init__(self):
        """
        Initializes the object and sets up our list of stopwords.
        """
        stop_words = set(stopwords.words('english'))
        self.stop_words = [s for s in stop_words if s not in ['no', 'nor', 'not']]  # I want negations

    def filter(self, spacy_token):
        """
        Token filter method. Returns true if the token she be included, false if the token should not be included.

        :param spacy_token: a spacy token
        :return: boolean
        """
        if not spacy_token.is_space \
           and not spacy_token.is_punct \
           and spacy_token.text.lower() not in self.stop_words \
           and not spacy_token.text.startswith('http:') \
           and not spacy_token.text.startswith('https:'):
            return True
        else:
            return False

    def is_stop_word(self, spacy_token):
        """
        Returns true if the spacy token is in this classes' list of stopwords, false otherwise.
        :param spacy_token: a spacy token
        :return: boolean
        """

        if spacy_token.text.lower() in self.stop_words:
            return True
        else:
            return False


class NounTokenFilter(TokenFilter):
    """
    Token filter that only matches nouns
    """
    def filter(self, spacy_token):
        """
        Token filter method. Returns true if the token she be included, false if the token should not be included.

        :param spacy_token: a spacy token
        :return: boolean
        """
        if spacy_token.pos_ == 'NOUN':
            return True
        else:
            return False
