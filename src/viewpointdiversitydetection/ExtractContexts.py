from nltk import SnowballStemmer

from viewpointdiversitydetection.ExtractedContextRanges import ExtractedContextRanges
from viewpointdiversitydetection.TermContext import TermContext
from viewpointdiversitydetection.TrackLeadingContext import TrackLeadingContext
from viewpointdiversitydetection.TrackTrailingContext import TrackTrailingContext


class ExtractContexts:
    def __init__(self, parseddocuments_obj, context_size, terms_to_extract):
        """
        Initialize the object, which will go ahead and extract contexts from the terms_to_extract
        parameter from the parsed documents found in parsedocuments_obj. Note that order is important here.
        The first key in the dictionary is extracted first, as is the first term in the first key. Subsequent terms
        can only be extracted if they do not fall in a segment that has already been extracted. So our contexts
        extracted per term will be weighted towards earlier terms in earlier contexts.

        :param parseddocuments_obj: An instance of ParsedDocuments which contains a Spacy parse of the corpus
        :param context_size: the number of valid tokens before and after the context to extract
        :param terms_to_extract: A dictionary with contexts to extract. The terms from each entry will be extracted
                                 at the same time. Of the form
                                 {'context_label':[term1, term2 ..], 'context_label2':[term1, term2, ..], ..}
        """
        self.ex = ExtractedContextRanges()  # object to keep track of what contexts we have extracted
        self.pd_obj = parseddocuments_obj
        self.stemmer = SnowballStemmer(language='english')  # create our own instance here
        self.context_size = context_size
        self.terms_to_extract = terms_to_extract
        self.contexts = {}  # contexts indexed by term to extract then the document index number
        # {context_label: {doc_id: [TermContext1, TermContext2, ..], ..}, context_label2: ...}

        for c in self.terms_to_extract.keys():
            self.contexts[c] = self._get_contexts(terms_to_extract[c])

    def get_contexts_by_doc_id_for(self, context_label):
        """
        Returns a dictionary with the contexts indexed by document id for the given context label.

        :param context_label: A string with label of the context that should be returned.
        """
        if context_label not in self.contexts:
            raise IndexError("%s is not a valid context label" % context_label)

        return self.contexts[context_label]

    def _get_contexts(self, terms):
        """
        Wrapper attribute whose main purpose is to use the stem to doc index to prepare a list
        of matching documents and then iterate through that list calling _get_contexts_for_multiple_terms
        for each matching document.

        :param terms: a list of terms for which to extract contexts from the corpus
        """

        # This is an optimization. The ParsedDocuments object provides access to a CorpusAsStems object
        # which has an index of stems and the documents they have been found in. We can quickly use this
        # to create a list of documents in the corpus which have the stem, so we are not having to scan
        # every document in the corpus on the off chance it contains the term we are trying to extract a
        # context for. As the number of terms we are searching for increases this is less and less helpful.
        matching_doc_indexes = []
        for t in terms:
            s = self.stemmer.stem(t)
            if s in self.pd_obj.corpusAsStems.stem_to_doc:  # it is a valid stem, as defined by the token filter
                # passed to the CorpusAsStems object
                matching_doc_indexes = matching_doc_indexes + self.pd_obj.corpusAsStems.stem_to_doc[s]

        matching_doc_indexes = list(set(matching_doc_indexes))  # make it unique
        matching_doc_indexes.sort()
        print("Searching %s documents which contain matching stems" % len(matching_doc_indexes))

        context_by_doc_index = {}
        # for doc_idx, doc in enumerate(all_docs):
        for doc_idx in matching_doc_indexes:
            doc = self.pd_obj.all_docs[doc_idx]
            context_by_doc_index[doc_idx] = self._get_contexts_for_multiple_terms(doc, doc_idx, terms)

        # we return a dictionary keyed by document index, each value is a list of TermContext objects
        # representing the contexts extracted from the document.
        return context_by_doc_index

    def _get_contexts_for_multiple_terms(self, document, document_index, match_terms):
        """
        Method which uses the collectors to get the contexts for the given terms. This
        is version two of the function. It can now efficiently extract contexts for multiple
        terms in a single sweep.

        Eventually I would like to remove the dependency on the Spacy methods for the
        internals of this function, but that isn't necessarily an immediate design goal.

        Returns a list of TermContext objects each containing contexts extracted from a matching term.

        :param document: A Spacy document
        :param document_index: The index number of the document we are processing, needed by the
                               extracted_contents object.
        :param match_terms: A list of terms that we are searching the document for
        """
        # Map our object variables into some local contexts, this made the port from the function easier :)
        context_size = self.context_size
        token_filter = self.pd_obj.token_filter
        stemmer = self.stemmer
        extracted_contents = self.ex

        match_stems = [stemmer.stem(t) for t in match_terms]
        leading_tokens = TrackLeadingContext(context_size, token_filter)
        trailing_tokens = TrackTrailingContext(context_size, token_filter)
        extracted_token_indexes = []
        token_index_to_stem = {}

        # First sweep through the tokens in the document and
        # grab the context of every word that matches our match_term
        for token_idx, token in enumerate(document):
            # if not token.is_space and not token.is_punct and token.text.lower() not in stop_words:
            # if token.pos_ == 'NOUN':
            if token_filter.filter(token):
                # Bug fix.
                # We moved the trailing_tokens.collect to below the creation of a new extraction
                # If it is above that occasionally a new term will match right on the last token
                # being extracted. If we call .collect first then .isCollecting() will be false,
                # although we did collect the token in question, and our extracted_contexts object
                # has not been updated, so we end up collected a duplicate context.
                # The extracted_contexts object ends up looking like this the below, which we don't want
                # [{'trigger_index': 1, 'start_index': 1, 'end_index': 7},
                #  {'trigger_index': 7, 'start_index': 1, 'end_index': 15},
                #  {'trigger_index': 15, 'start_index': 7, 'end_index': 19}]
                was_collected = False
                if trailing_tokens.is_collecting():
                    was_collected = True

                trailing_tokens.collect(token, token_idx)
                # now see if this new token matches our match_term, if so
                # we 'trigger' a collection of leading and trailing context
                #
                # We don't do that if the current token is on a range that has already been extracted, or if we
                # are currently collecting. If we are currently collecting that means that we already had a match
                # but haven't gathered enough context for it. Since this token will be extracted anyways no need
                # to see if it matches any other terms.
                #
                # Not that TrackTrailingContext is designed to be able to collect multiple contexts at the same time.
                # By not attempting a match when it is still collecting from the last match we are constraining the
                # algorithm to collect only one context at a time.
                if not extracted_contents.has_been_extracted(document_index, token_idx) and not was_collected:
                    token_stem = stemmer.stem(token.text)
                    if token_stem in match_stems:
                        # It matches, take a snapshot of the context we have seen
                        leading_tokens.remember_last_tokens(token_idx)
                        # and spawn a new collector to grab the next x tokens of context
                        trailing_tokens.remember_next_tokens(token_idx)
                        # increment our trigger index
                        extracted_token_indexes.append(token_idx)
                        # Keep track of which stem this context belongs to
                        token_index_to_stem[token_idx] = token_stem

                # keep track of the leading context
                leading_tokens.collect(token, token_idx)

        # Construct our context objects and update the extracted_contents object

        # each of these is a dictionary like the following
        # {trigger_index_1:['string','string','string',..], trigger_index_2:['string',...], ...}
        leading_by_trigger = leading_tokens.return_leading_contexts()
        trailing_by_trigger = trailing_tokens.return_trailing_contexts()
        # We create a context object per stem we are looking for
        # so that we can gatherup the leading and trailing contexts
        # context_objs = TermContext(match_term)
        context_objs = {}
        for ms in match_stems:
            context_objs[ms] = TermContext(ms)

        for t in extracted_token_indexes:
            stem = token_index_to_stem[t]  # the trigger stem, this is what matched
            leading = []
            trailing = []
            leading_index = -1
            trailing_index = -1
            if t in leading_by_trigger:
                leading = leading_by_trigger[t]['token_text_list']
                leading_index = leading_by_trigger[t]['starting_index']
            if t in trailing_by_trigger:
                trailing = trailing_by_trigger[t]['token_text_list']
                trailing_index = trailing_by_trigger[t]['ending_index']
            if len(leading) != 0 or len(trailing) != 0:
                context_objs[stem].add_context(leading, trailing)
                extracted_contents.record_extraction(document_index, t, leading_index, trailing_index)

        # Return a list of TermContext objects, each having all the contexts for each of the matching terms
        # in the document. But only if we actually grabbed any context for that term.
        return [co for co in context_objs.values() if co.has_context()]
