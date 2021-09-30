from viewpointdiversitydetection.CollectXTokens import CollectXTokens


class TrackTrailingContext:
    """
    Fixes an annoying interface difference between the trailing context, which
    is processed as a list of listeners and the leading context, which is captured
    continually and just "snapshotted" when we have a match. This wrapper allows
    the trailing context to be treated more like the leading context. It standardizes
    the collection of the tokens and also the datastructure that is returned when we
    finish processing the document.

    :param number_of_tokens: The number of token the object should collect before ignoring additional tokens
    :param token_filter: a function that takes a Spacy token (class 'spacy.tokens.token.Token) and returns True if it
                         should be return
    """

    def __init__(self, number_of_tokens, token_filter):
        self.collectors = []
        self.number_of_tokens = number_of_tokens
        self.token_filter = token_filter

    def collect(self, token, token_index):
        """
        Offer the token to each collector we have spawned
        if it wants it, it will take it, if it already has
        enough tokens it will ignore it.

        :param token: a Spacy token to collect
        :type token: class 'spacy.tokens.token.Token'
        :param token_index: integer representing where the token was in the original document
        """
        # print("Collect called, we have %s collectors." % len(self.collectors))

        for c in self.collectors:
            # print("Offering token %s to collectors." % token.text)
            c.collect(token, token_index)

    def is_collecting(self):
        """
        Returns true if are tracking any collectors that still do not have enough tokens
        False otherwise.
        """
        for c in self.collectors:
            if c.is_collecting():
                return True
        return False

    def remember_next_tokens(self, trigger):
        """
        Create a new CollectXTokens object so that we can remember the next few
        tokens after the trigger.

        :param trigger: a string or number used to identify why these tokens are being collected, it is returned with
                        the list of strings
        """
        #
        # print("Creating new listener for trigger %s" % trigger)
        self.collectors.append(CollectXTokens(self.number_of_tokens, self.token_filter, trigger))
        # print("We now have %s collectors." % len(self.collectors))

    def return_trailing_contexts(self):
        """
        Returns a dictionary with the trigger and the list
        of tokens. Create a list with this structure for each collector we have
        and then return it.
        """
        triggers_and_tokens = [ct.get_tokens_as_text_list() for ct in self.collectors]
        tokens_by_trigger = {}
        for d in triggers_and_tokens:
            # tokens_by_trigger[d['trigger']] = d['token_text_list']
            tokens_by_trigger[d['trigger']] = d
        return tokens_by_trigger
