from nltk import SnowballStemmer
from nltk.corpus import stopwords

from viewpointdiversitydetection.TokenFilter import NounTokenFilter
from viewpointdiversitydetection.ExtractedContextRanges import ExtractedContextRanges
from viewpointdiversitydetection.TermContext import TermContext
from viewpointdiversitydetection.TrackLeadingContext import TrackLeadingContext
from viewpointdiversitydetection.TrackTrailingContext import TrackTrailingContext


class FindCharacteristicKeywords:

    def __init__(self, parsed_document_object):
        """
        Initialize the Object

        :param parsed_document_object: a ParsedDocumentsFourForums object
        """
        # Initialize the extracted context ranges object, this will keep track of the range of every
        # context we extract from the corpus
        self.extracted_contexts = ExtractedContextRanges()
        self.stemmer = SnowballStemmer(language='english')
        stop_words = set(stopwords.words('english'))
        self.stop_words = [s for s in stop_words if s not in ['no', 'nor', 'not']]  # I want negations
        self.pdo = parsed_document_object
        if len(self.pdo.all_docs) == 0:
            raise ValueError("ParsedDocument object contains no parsed documents.")

    def _get_context_for_multiple_terms(self, document, document_index, match_terms, context_size,
                                        token_filter, context_label):
        """
        Function which uses the collectors to get the contexts for the given terms. This
        is version two of the function and it can efficiently extract contexts for multiple
        terms in a single sweep.

        Eventually I would like to remove the dependency on the Spacy methods for the
        internals of this function, but that isn't necessarily a immediate design goal.

        :param document: A Spacy document
        :param document_index: The index number of the document we are processing, needed by the
                               extracted_contents object.
        :param match_terms: A list of terms that we are searching the document for
        :param context_size: The number of tokens leading and trailing context we are collecting
        :param token_filter: A function that returns a boolean indicating if a certain taken should be included
                                   in the context. Note that this does not affect the number of tokens we collect, it
                                   is only applied when we actually return the context. Put another way token_filter
                                   is applied after 'content_size' tokens have been collected and before the contexts
                                   are returned.
        :param context_label: the label for the contexts we are extracting
        """
        stemmer = self.stemmer
        extracted_contents = self.extracted_contexts
        stop_words = self.stop_words

        match_stems = [stemmer.stem(t) for t in match_terms]
        leading_tokens = TrackLeadingContext(context_size, token_filter)
        trailing_tokens = TrackTrailingContext(context_size, token_filter)
        extracted_token_indexes = []
        token_index_to_stem = {}

        # First sweep through the tokens in the document and
        # grab the context of every word that matches our match_term
        for token_idx, token in enumerate(document):
            if not token.is_space and not token.is_punct and token.text.lower() not in stop_words:
                # keep track of the trailing context
                trailing_tokens.collect(token, token_idx)

                # now see if this new token matches our match_term, if so
                # we 'trigger' a collection of leading and trailing context
                #
                # We don't do that if the current token is on a range that has already been extracted, or if we
                # are currently collecting. If we are currently collecting that means that we already had a match
                # but haven't gathered enough context for it. Since this token will be extracted anyways no need
                # to see if it matches any other terms.
                #
                # Not that TrackTrailingContext is designed to be able to collect multiple contexts at the same time
                # by not attempting a match when it is still collecting from the last match we constraining the
                # algorithm to collect only one context at a time.
                if not extracted_contents.has_been_extracted(document_index, token_idx) \
                   and not trailing_tokens.is_collecting():
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
        # so that we can gather up the leading and trailing contexts
        context_objects = {}
        for ms in match_stems:
            context_objects[ms] = TermContext(ms)

        for t in extracted_token_indexes:
            stem = token_index_to_stem[t]  # the trigger stem, this is what matched
            leading = []
            leading_index = 0
            trailing_index = 0
            trailing = []
            if t in leading_by_trigger:
                leading = leading_by_trigger[t]['token_text_list']
                leading_index = leading_by_trigger[t]['starting_index']
            if t in trailing_by_trigger:
                trailing = trailing_by_trigger[t]['token_text_list']
                trailing_index = trailing_by_trigger[t]['ending_index']
            if len(leading) != 0 or len(trailing) != 0:
                context_objects[stem].add_context(leading, trailing)
                extracted_contents.record_extraction(document_index, t, leading_index, trailing_index, context_label)

        # Return a list of TermContext objects, each having all the contexts for each of the matching terms
        # in the document. But only if we actually grabbed any context for that term.
        return [co for co in context_objects.values() if co.has_context()]

    def get_unique_nouns_from_term_context(self, terms, context_label):
        """

        :param terms: the list of terms to extract the contexts for
        :param context_label: the label for the contexts we are extracting
        :return:
        """
        # Grab the stem to doc index and the list of all parsed documents
        # from the ParsedDocuments object
        stem_to_doc = self.pdo.corpusAsStems.stem_to_doc
        all_docs = self.pdo.all_docs
        stemmer = self.stemmer
        stop_words = self.stop_words

        noun_token_filter = NounTokenFilter()

        def get_nouns(context_dict):
            leading_nouns = [noun for noun in context_dict['leading_tokens']]
            trailing_nouns = [noun for noun in context_dict['trailing_tokens']]
            return leading_nouns + trailing_nouns

        matching_doc_indexes = []
        for t in terms:
            s = stemmer.stem(t)
            if s in stem_to_doc and s not in stop_words:
                # if the term is a stop word for example the string 'ins' used a noun
                # then we don't consider it.
                matching_doc_indexes = matching_doc_indexes + stem_to_doc[s]
        matching_doc_indexes = list(set(matching_doc_indexes))  # make it unique
        matching_doc_indexes.sort()
        print("Searching %s documents which contain matching stems" % len(matching_doc_indexes))

        all_nouns = []
        for doc_idx in matching_doc_indexes:
            doc = all_docs[doc_idx]

            contexts = self._get_context_for_multiple_terms(doc, doc_idx, terms, 4, noun_token_filter, context_label)
            for context in contexts:
                nouns_from_contexts = []
                for n in context.contexts:
                    nouns_from_contexts = nouns_from_contexts + get_nouns(n)
                nouns_from_contexts = list(set(nouns_from_contexts))
                all_nouns = all_nouns + nouns_from_contexts

        unique_nouns = list(set(all_nouns))
        self.analyze(unique_nouns)
        return unique_nouns

    def analyze(self, unique_nouns):
        ex = self.extracted_contexts
        all_docs = self.pdo.all_docs

        print("Performed %s total extractions" % len(ex.extractions))
        print("We have extracted %s new unique nouns" % len(unique_nouns))

        documents_with_extractions = list(ex.extractions.keys())
        print("Extracted from %s total documents" % len(documents_with_extractions))
        coverage = len(documents_with_extractions) / len(all_docs)
        print("Extraction coverage: %.4f percent" % coverage)
