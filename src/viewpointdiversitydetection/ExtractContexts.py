from nltk import SnowballStemmer

from viewpointdiversitydetection.ExtractedContextRanges import ExtractedContextRanges
from viewpointdiversitydetection.TermContext import TermContext
from viewpointdiversitydetection.TrackLeadingContext import TrackLeadingContext
from viewpointdiversitydetection.TrackTrailingContext import TrackTrailingContext


class ExtractContexts:
    def __init__(self, parseddocuments_obj, context_size, token_filter=None):
        """
        Initialize the object, which will go ahead and extract contexts from the terms_to_extract
        parameter from the parsed documents found in parsedocuments_obj. Note that order is important here.
        The first key in the dictionary is extracted first, as is the first term in the first key. Subsequent terms
        can only be extracted if they do not fall in a segment that has already been extracted. So our contexts
        extracted per term will be weighted towards earlier terms in earlier contexts.

        :param parseddocuments_obj: An instance of ParsedDocuments which contains a Spacy parse of the corpus
        :param context_size: the number of valid tokens before and after the context to extract
        :param token_filter: A function that returns a boolean indicating if a certain taken should be included
                                   in the context. Note that this does not affect the number of tokens we collect, it
                                   is only applied when we actually return the context. Put another way token_filter
                                   is applied after 'content_size' tokens have been collected and before the contexts
                                   are returned. If no token_filter is passed then the object will use the token filter
                                   that was used in parsing the corpus, parseddocuments_obj.token_filter
        """
        self.ex = ExtractedContextRanges()  # object to keep track of what contexts we have extracted
        self.pd_obj = parseddocuments_obj
        self.stemmer = SnowballStemmer(language='english')  # create our own instance here
        self.context_size = context_size
        self.terms_to_extract = None
        self.contexts = {}  # contexts indexed by term to extract then the document index number
        # {context_label: {doc_id: [TermContext1, TermContext2, ..], ..}, context_label2: ...}
        if token_filter:
            self.token_filter = token_filter
        else:
            self.token_filter = parseddocuments_obj.token_filter

    def extract_contexts(self, terms_to_extract):
        """
        Perform the actual extraction.

        :param terms_to_extract: A dictionary with contexts to extract. The terms from each entry will be extracted
                         at the same time. Of the form
                         {'context_label':[term1, term2 ..], 'context_label2':[term1, term2, ..], ..}
        :return: None, the extracted contexts are stored in the 'contexts' object variable.
        """
        self.terms_to_extract = terms_to_extract
        for c in self.terms_to_extract.keys():
            self.contexts[c] = self._get_contexts(terms_to_extract[c], c)

    def extract_additional_contexts(self, terms_to_extract):
        """
        Extracts additional contexts using additional terms. The method adds to the contexts extracted
        and will not re-extract existing contexts over overwrite previous contexts.

        :param terms_to_extract: A dictionary with contexts to extract. The terms from each entry will be extracted
                         at the same time. Of the form
                         {'context_label':[term1, term2 ..], 'context_label2':[term1, term2, ..], ..}
        :return: None, the new contexts are added to the existing 'contexts' object variable
        """
        if not self.terms_to_extract:
            # We have never done an initial extract, so do that instead
            # to initialize the data structures
            self.extract_contexts(terms_to_extract)
        else:
            # This is really an additional extraction, proceed

            # Keep track of the total terms that we have processed by adding the new terms
            # to the self.terms_to_extract structure by context label
            # We also want to filter out terms that we have already extracted!
            filtered_terms_to_extract = {label: [] for label in terms_to_extract.keys()}
            for label, new_terms in terms_to_extract.items():
                if label in self.terms_to_extract:
                    for t in new_terms:  # iterate from new terms to preserve order!
                        if t not in self.terms_to_extract[label]:  # we haven't extracted this term before in this context
                            self.terms_to_extract[label].append(t)
                            filtered_terms_to_extract[label].append(t)
                else:  # totally new context, definitely haven't seen it before
                    self.terms_to_extract[label] = new_terms
                    filtered_terms_to_extract[label] = new_terms

            # Extract the contexts for the new terms and
            # update the self.contexts data structure
            for label, terms in filtered_terms_to_extract.items():  # for each label
                # new_contexts contains context_by_doc_idx: the list of TermContext objects indexed by document id
                # {doc_idx1: [TermContext1, TermContext2, ...], doc_idx2: [TermContext10, ...], ...}
                if len(terms) > 0:  # we have terms to extract!
                    new_contexts = self._get_contexts(terms, label)  # extract the new contexts

                    for doc_idx, term_contexts in new_contexts.items():  # for every set of new contexts
                        if label not in self.contexts:
                            self.contexts[label] = {}  # we haven't seen this context label before, create it
                        if doc_idx in self.contexts[label]:  # if we already have contexts for this document
                            # {context_label: {doc_id: [TermContext1, TermContext2, ..], ..}, context_label2: ...}
                            self.contexts[label][doc_idx] += term_contexts  # add the new contexts
                        else:  # we haven't had contexts for this document before
                            self.contexts[label][doc_idx] = term_contexts

    def get_contexts_by_doc_id_for(self, context_label):
        """
        Returns a dictionary with the contexts indexed by document id for the given context label.

        :param context_label: A string with label of the context that should be returned.
        """
        if context_label not in self.contexts:
            raise IndexError("%s is not a valid context label" % context_label)

        return self.contexts[context_label]

    def _get_contexts(self, terms, context_label):
        """
        Wrapper attribute whose main purpose is to use the stem to doc index to prepare a list
        of matching documents and then iterate through that list calling _get_contexts_for_multiple_terms
        for each matching document.

        :param terms: a list of terms for which to extract contexts from the corpus
        :param context_label: the label of the context that these terms are from
        :return context_by_doc_idx: the list of TermContext objects indexed by document id
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
        # print("Searching %s documents which contain matching stems" % len(matching_doc_indexes))

        context_by_doc_index = {}
        # for doc_idx, doc in enumerate(all_docs):
        for doc_idx in matching_doc_indexes:
            doc = self.pd_obj.all_docs[doc_idx]
            context_by_doc_index[doc_idx] = self._get_contexts_for_multiple_terms(doc, doc_idx, terms, context_label)

        # we return a dictionary keyed by document index, each value is a list of TermContext objects
        # representing the contexts extracted from the document.
        return context_by_doc_index

    def get_contexts_for_multiple_terms(self, document, document_index, match_terms, context_label):
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
        :param context_label: the label of the context that these terms are from
        :return: Return a list of TermContext objects
        """
        return self._get_contexts_for_multiple_terms(document, document_index, match_terms, context_label)

    def _get_contexts_for_multiple_terms(self, document, document_index, match_terms, context_label):
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
        :param context_label: the label of the context that these terms are from
        :return: Return a list of TermContext objects
        """
        # Map our object variables into some local contexts, this made the port from the function easier :)
        context_size = self.context_size
        token_filter = self.token_filter
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
                context_objs[stem].add_context_with_indices(leading, trailing, leading_index, trailing_index,
                                                            self.get_sentence_indexes_from_token_range(t,
                                                                                                       leading_index,
                                                                                                       trailing_index,
                                                                                                       document)
                                                            )
                extracted_contents.record_extraction(document_index, t, leading_index, trailing_index, context_label)

        # Return a list of TermContext objects, each having all the contexts for each of the matching terms
        # in the document. But only if we actually grabbed any context for that term.
        return [co for co in context_objs.values() if co.has_context()]

    def get_sentence_indexes_from_token_range(self, token_idx, start_doc_token_idx, end_doc_token_idx, doc):
        """
        Given a beginning and ending token this function uses the Spacy parsed
        document to find and return the sentences that contain that range of
        tokens.

        :param start_doc_token_idx: the token index that marks the beginning of the range
        :param end_doc_token_idx:  the token index that marks the end of the range
        :param doc: the document as a Spacy object
        :return: A list of spacy sentences
        """
        # We check for None because 0 is a valid index
        if start_doc_token_idx is None:
            # A match at the start of the document, so the start token index is the token index of the trigger word
            start_doc_token_idx = token_idx
        elif end_doc_token_idx is None:
            # A match at the end of the document, so the end token index is the token index of the trigger word
            end_doc_token_idx = token_idx
        if start_doc_token_idx >= end_doc_token_idx:
            raise RuntimeError("The ending token index cannot be greater than or equal to the starting token index")
        start_sentence_idx = 0
        end_sentence_idx = 0
        doc_token_idx = 0
        for i, s in enumerate(doc.sents):
            for t in s:
                # t contains the token, but we ignore that, we are only interested in the token index in document
                # which is being kept track of in doc_token_index
                if doc_token_idx == start_doc_token_idx:
                    start_sentence_idx = i
                # The starting token index and ending token index can be the same token
                # so this is an 'if' not an 'elsif'.
                if doc_token_idx == end_doc_token_idx - 1:
                    end_sentence_idx = i
                doc_token_idx += 1
        # print(start_sentence_idx, end_sentence_idx)
        return list(range(start_sentence_idx, end_sentence_idx + 1))

    def get_sentences_and_terms_by_doc_idx(self, doc_idx, include_all_data=False):
        """
        Get summary data, and optionally the raw token and sentence data, from the contexts extracted
        for a specific document id. Summary fields are:
            total_terms - total number of tokens in the entire document
            terms_matching - number of tokens that were matched
            terms_extracted - total number of tokens extracted because of those matches
            unique_terms_extracted - the number of unique tokens extracted
            total_sentences - total number of sentences in the document according to Spacy
            sentences_extracted - number of sentences extracted to encompass the extracted terms
            percent_terms_extracted - percent_terms_extracted,
            percent_sentences_extracted - percent_sentences_extracted

        And the data fields are as follows if include_al/_data is True
            sentences - all extracted sentences
            tokens - all_tokens_extracted

        :param doc_idx: the document index
        :param include_all_data: boolean, include extracted sentences and tokens, defaults to False
        :return: dictionary of fields and data as described above
        """
        # For a given document index, get all the matching search terms
        # as well as all the text sentences that matched those terms
        all_labels = self.contexts.keys()
        term_contexts = []
        for l in all_labels:
            if doc_idx in self.contexts[l]:
                term_contexts = term_contexts + self.contexts[l][doc_idx]
        matching_terms = [tc.term for tc in term_contexts]

        # Get sentence indices for all contexts
        sentence_indices_list = []
        tokens_extracted = 0
        all_tokens_extracted = []
        for tc in term_contexts:
            for indices in tc.sentence_indices:
                sentence_indices_list += indices
            for c in tc.contexts:
                tokens_extracted += len(c['leading_tokens'])
                all_tokens_extracted += c['leading_tokens']
                tokens_extracted += len(c['trailing_tokens'])
                all_tokens_extracted += c['trailing_tokens']
        unique_tokens_extracted = len(set(all_tokens_extracted))

        total_tokens = len(self.pd_obj.all_docs[doc_idx])
        sentence_indices = list(set(sentence_indices_list))
        sentences = []
        total_sentences = 0
        for i, sent in enumerate(self.pdo_obj.all_docs[doc_idx].sents):
            total_sentences += 1
            if i in sentence_indices:
                sentences.append(sent.text)

        if total_tokens > 0:
            percent_terms_extracted = tokens_extracted * 1.0 / total_tokens
        else:
            percent_terms_extracted = 0

        if total_sentences > 0:
            percent_sentences_extracted = len(sentences) * 1.0 / total_sentences
        else:
            percent_sentences_extracted = 0

        data = {'total_terms': total_tokens,  # total number of tokens in the entire document
                'terms_matching': len(matching_terms),  # number of tokens that were matched
                'terms_extracted': tokens_extracted,  # total number of tokens extracted because of those matches
                'unique_terms_extracted': unique_tokens_extracted,  # the number of unique tokens extracted
                'total_sentences': total_sentences,  # total number of sentences in the document according to Spacy
                'sentences_extracted': len(sentences),  # number of sentences extracted to encompass the extracted terms
                'percent_terms_extracted': percent_terms_extracted,
                'percent_sentences_extracted': percent_sentences_extracted
                }
        if include_all_data:
            data['sentences'] = sentences
            data['tokens'] = all_tokens_extracted

        return data
