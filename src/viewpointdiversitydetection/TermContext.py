class TermContext:
    """
    Class to hold the context of a term including the index of the sentences from which
    the content was extracted.
    It is a bit involved and I don't want to keep track of the whole thing with raw
    lists of dictionaries of lists.

    :param term: the term we are grabbing context for
    """

    def __init__(self, term):
        self.term = term  # the term we are grabbing context for
        self.contexts = []  # list of dictionaries containing the leading and trailing tokens
        self.sentence_indices = []  # list of lists of sentence indices, one for each context
        self.token_indices = []  # list of lists of starting and ending token indexes for this context

    # Returns True if the object contains at least one context
    # False otherwise.
    def has_context(self):
        """
        Returns True if the object contains at least one context, False otherwise.
        """
        if len(self.contexts) == 0:
            return False
        else:
            return True

    # Adds the context from a document
    def add_context(self, before, after):
        """
        Our context structure, the document index, and the tokens that came before
        and the tokens that came after. This method does not store the sentence indices or the beginning
        and ending token indices.

        :param before: a list of strings representing the leading context of the term
        :param after: a list of strings representing the trailing context of the term
        """
        context_structure = {'leading_tokens': before,
                             'trailing_tokens': after}

        # A list with all the contexts in it
        self.contexts.append(context_structure)
        self.sentence_indices.append([])
        self.token_indices.append([])

    def add_context_with_indices(self, before, after, starting_index, ending_index, sentence_indices):
        """
        Add the context along with the token and sentence indices. This includes the index of the first and last
        token in the context as well as the sentence indices that indicate the sentences in the document that the context
        was extracted from.

        :param before: a list of strings representing the leading context of the term
        :param after: a list of strings representing the trailing context of the term
        :param starting_index: The token index at which this specific context began
        :param ending_index: The token index at which this specific content ended (i.e. the index of the last token)
        :param sentence_indices: the sentence indices as a list of integers
        """
        context_structure = {'leading_tokens': before,
                             'trailing_tokens': after}
        token_index_structure = {'starting_index': starting_index,
                                 'ending_index': ending_index}

        # A list with all the contexts in it
        self.contexts.append(context_structure)
        self.sentence_indices.append(sentence_indices)
        self.token_indices.append(token_index_structure)