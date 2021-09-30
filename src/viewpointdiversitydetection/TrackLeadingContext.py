class TrackLeadingContext:
    """
    Class that remembers the last x tokens it has collected. Upon calling the rememberLastTokens method
    it will save off the the last 'number_of_tokens' tokens along with an identifying trigger. It can then
    return all of the saved lists along with the identifying triggers.

    :param number_of_tokens: The number of token the object should collect before ignoring additional tokens
    :param token_filter: a function that takes a Spacy token (class 'spacy.tokens.token.Token) and returns True if
                         it should be return
    """

    def __init__(self, number_of_tokens, token_filter):
        self.number_of_tokens = number_of_tokens
        self.token_filter = token_filter
        self.tokens = []
        self.token_indexes = []
        self.snapshots = []
        self.triggers = []  # a list of strings, these are the tokens that caused us to be
        # initialized in the first place. This is the 'why' we are collecting.
        # useful for algorithm audit
        self.indexes = []  # holds the starting index and ending index for each trigger as a tuple

    def collect(self, token, token_index):
        """
        Collect a new token, if we now have too many then drop the oldest token.

        :param token: a Spacy token
        :param token_index: integer representing where the token was in the original document
        """
        self.tokens.append(token)
        self.token_indexes.append(token_index)
        if len(self.tokens) > self.number_of_tokens:
            self.tokens.pop(0)
            self.token_indexes.pop(0)

    def remember_last_tokens(self, trigger):
        """
        Save the last 'number_of_tokens' tokens so that we can remember the tokens that came before the trigger.

        :param trigger: a string or number used to identify why these tokens are being collected, it is returned with
                        the list of strings
        """
        # Save the tokens we are remembering as a list of strings
        if len(self.tokens) == 0:
            # Perhaps the word occurs at the beginning of the sentence
            # print("Warning: We do not have any leading tokens for trigger %s, snapshot empty." % trigger)
            self.snapshots.append([])
            self.triggers.append(trigger)
            self.indexes.append((None, None))  # empty tuple, no start or end indexes because we don't have any tokens
        else:
            token_texts = [t.text for t in self.tokens if self.token_filter(t)]
            self.snapshots.append(token_texts)
            self.triggers.append(trigger)
            self.indexes.append((self.token_indexes[0],
                                 self.token_indexes[-1]))  # the start and end index for the tokens in self.tokens

    def return_leading_contexts(self):
        """
        Returns a dictionary with the trigger and the list
        of tokens.
        """
        triggers_and_tokens = [{'trigger': i[0], 'token_text_list': i[1], 'indexes': i[2]} for i in
                               zip(self.triggers, self.snapshots, self.indexes)]
        tokens_by_trigger = {}
        for d in triggers_and_tokens:
            tokens_by_trigger[d['trigger']] = {'token_text_list': d['token_text_list'],
                                               'starting_index': d['indexes'][0], 'ending_index': d['indexes'][1]}
        return tokens_by_trigger
