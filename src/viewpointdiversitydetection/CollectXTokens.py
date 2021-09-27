class CollectXTokens:
    """
    Class that will collect a specified number of tokens and then return specific tokens as
    a list of strings.

    :param number_of_tokens: The number of token the object should collect before ignoring additional tokens
    :param token_filter: a function that takes a Spacy token (class 'spacy.tokens.token.Token) and returns True if it
                         should be return
    :trigger: a string or number used to identify why these tokens are being collected, it is returned with the list
              of strings
    """

    def __init__(self, number_of_tokens, token_filter, trigger):
        self.number_of_tokens = number_of_tokens
        self.token_filter = token_filter
        self.trigger = trigger  # a list of identifiers, these are the tokens that caused us to be
        # initialized in the first place. This is the 'why' we are collecting.
        # useful for algorithm audit and matching leading and trailing context
        self.tokens = []
        self.token_indexes = []
        self.tokens_collected = 0

    def collect(self, token, token_index):
        """
        Collect a new token if we do not have enough tokens. If we already have enough tokens just ignore
        the new offering.

        :param token: a Spacy token to collect
        :type token: class 'spacy.tokens.token.Token'
        :param token_index: integer representing where the token was in the original document
        """
        if self.tokens_collected < self.number_of_tokens:
            self.tokens.append(token)
            self.token_indexes.append(token_index)
        self.tokens_collected = self.tokens_collected + 1

    def is_collecting(self):
        """
        Return true if we have enough tokens. False otherwise.
        """
        if self.tokens_collected < self.number_of_tokens:
            return True
        return False

    def get_tokens_as_text_list(self):
        """
        Returns the token we have collected as a list. This is where the token_filter
        specified on object creation is applied. The object will have collected up to
        'number_of_tokens' tokens but will only return tokens where
        token_filter(token) == True
        """
        token_texts = [t.text for t in self.tokens if self.token_filter(t)]
        if len(self.token_indexes) > 0:
            starting_index = self.token_indexes[0]
            ending_index = self.token_indexes[-1]
        else:  # we didn't end up collecting any context, we have no tokens
            starting_index = None
            ending_index = None

        return ({'trigger': self.trigger, 'token_text_list': token_texts, 'starting_index': starting_index,
                 'ending_index': ending_index})
