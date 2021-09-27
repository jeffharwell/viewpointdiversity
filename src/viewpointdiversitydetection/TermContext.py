class TermContext:
    """
    Class to hold the context of a term.
    It is a bit involved and I don't want to keep track of the whole thing with raw
    lists of dictionaries of lists.

    :param term: the term we are grabbing context for
    """

    def __init__(self, term):
        self.term = term  # the term we are grabbing context for
        self.contexts = []

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

    # Adds the a context from a document
    def add_context(self, before, after):
        """
        Our context structure, the document index, and the tokens that came before
        and the tokens that came after.

        :param before: a list of strings representing the leading context of the term
        :param after: a list of strings representing the trailing context of the term
        """
        context_structure = {'leading_tokens': before,
                             'trailing_tokens': after}

        # A list with all the contexts in it
        self.contexts.append(context_structure)
