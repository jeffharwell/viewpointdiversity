class CorpusDefinition:
    """
    All the information that we need to create process a
    corpus. Handy for batch processing corpora.
    """

    def __init__(self, topic):
        """
        Initialize the object
        :param topic: string with the name of the topic
        """
        self.topic = topic
        self.stance_a = None
        self.stance_b = None
        self.search_terms = None
        self.transform = None

    def set_stance(self, stance_a, stance_b):
        """
        Set the stances.

        :param stance_a: string, text of stance a
        :param stance_b: string, text of stance b
        """

        self.stance_a = stance_a
        self.stance_b = stance_b

    def set_search_terms(self, search_terms):
        """
        Set the search terms to use when extracting the contexts.

        :param search_terms: the search terms to use
        """
        self.search_terms = search_terms

    def set_transform(self, transform_function):
        """
        Set the function that should be used by the PDO to transform the raw
        text before parsing.

        :param transform_function: function which takes a string as an input and outputs a string
        """
        self.transform = transform_function