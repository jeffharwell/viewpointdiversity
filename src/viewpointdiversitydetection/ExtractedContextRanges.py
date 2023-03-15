class ExtractedContextRanges:
    """
    A class to keep track of the range of tokens that we have already extracted.
    """

    def __init__(self):
        self.extractions = {}

    def record_extraction(self, doc_index, trigger_index, start_index, end_index, context_label):
        """
        Record an extraction.

        :param doc_index: the document we extracted the context from
        :param trigger_index: the index of the word in the document that triggered the context extraction
        :param start_index: the token index where the context extraction started (inclusive)
        :param end_index: the token index where the context extraction ended (inclusive)
        """
        # Our start or end indexes might be None. This means that there were not tokens proceed or
        # following the trigger token respectively. In this case the beginning or ending of our extraction
        # range in the trigger index itself.
        #
        # A single word document that had a matching keyword would end up with the following
        # extraction record:
        # {'trigger_index':0, 'start_index':0, 'end_index':0}
        if not isinstance(doc_index, int):
            raise RuntimeError("Document index must be of type int")
        if not isinstance(trigger_index, int):
            raise RuntimeError("Trigger index must be of type int")
        if not isinstance(start_index, int) and start_index is not None:
            print("Received Start Index: %s" % start_index)
            raise RuntimeError("Start index must be of type int.")
        if not isinstance(end_index, int) and end_index is not None:
            print("Received End Index: %s" % end_index)
            raise RuntimeError("End index must be of type int.")

        if not start_index:  # it might be None
            start_index = trigger_index
        if not end_index:  # also might be None
            end_index = trigger_index

        e = {'trigger_index': trigger_index, 'start_index': start_index,
             'end_index': end_index, 'context_label': context_label}

        if doc_index in self.extractions:
            self.extractions[doc_index].append(e)
        else:
            self.extractions[doc_index] = [e]

    def has_been_extracted(self, doc_index, trigger_index):
        """
        Returns True if this trigger is from a context that has already been extracted,
        False otherwise.

        :param doc_index: the index of the document we are checking
        :param trigger_index: the index of the token we are checking
        """
        if doc_index in self.extractions:
            for e in self.extractions[doc_index]:
                # our start and end indexes are inclusive, so >= and <=
                if e['start_index'] <= trigger_index <= e['end_index']:
                    return True
        return False

    def get_number_of_extractions(self):
        """
        Returns the number of extractions that have been recorded across all documents.
        """
        extraction_count = 0
        for extractions in self.extractions.values():  # the list of extractions for all documents
            extraction_count = extraction_count + len(extractions)
        return extraction_count
