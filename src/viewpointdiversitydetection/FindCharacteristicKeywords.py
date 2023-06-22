from nltk import SnowballStemmer
from nltk.corpus import stopwords

from viewpointdiversitydetection.TokenFilter import NounTokenFilter
from viewpointdiversitydetection.ExtractedContextRanges import ExtractedContextRanges
from viewpointdiversitydetection.TermContext import TermContext
from viewpointdiversitydetection.ExtractContexts import ExtractContexts
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
        self.extracted_contexts = None
        self.stemmer = SnowballStemmer(language='english')
        stop_words = set(stopwords.words('english'))
        self.stop_words = [s for s in stop_words if s not in ['no', 'nor', 'not']]  # I want negations
        self.pdo = parsed_document_object
        if len(self.pdo.all_docs) == 0:
            raise ValueError("ParsedDocument object contains no parsed documents.")

    def get_unique_nouns_from_term_context(self, terms, context_label, context_size=4):
        """

        :param terms: the list of terms to extract the contexts for
        :param context_label: the label for the contexts we are extracting
        :param context_size: the number of tokens of context to consider when extracting nouns, defaults to 4
        :return:
        """
        # Grab the stem to doc index and the list of all parsed documents
        # from the ParsedDocuments object
        stem_to_doc = self.pdo.corpusAsStems.stem_to_doc
        all_docs = self.pdo.all_docs
        stemmer = self.stemmer
        stop_words = self.stop_words

        noun_token_filter = NounTokenFilter()
        # We will use the ExtractContexts logic, but we only want to extract from
        # certain documents, so initialize the object here and use the context extraction
        # method below in the loop over the matching documents.
        ec = ExtractContexts(self.pdo, context_size, noun_token_filter)

        def get_nouns(context_dict):
            leading_nouns = [noun for noun in context_dict['leading_tokens']]
            trailing_nouns = [noun for noun in context_dict['trailing_tokens']]
            return leading_nouns + trailing_nouns

        #
        # First create a list of matching documents
        # so that we don't spend time running the context extraction against documents
        # that have no match to our terms.
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

        # Now go through each of the matching documents and extract the contexts
        all_nouns = []
        for doc_idx in matching_doc_indexes:
            doc = all_docs[doc_idx]

            # contexts = self._get_context_for_multiple_terms(doc, doc_idx, terms, 4, noun_token_filter, context_label)
            contexts = ec.get_contexts_for_multiple_terms(doc, doc_idx, terms, context_label)
            for context in contexts:
                nouns_from_contexts = []
                for n in context.contexts:
                    nouns_from_contexts = nouns_from_contexts + get_nouns(n)
                nouns_from_contexts = list(set(nouns_from_contexts))
                all_nouns = all_nouns + nouns_from_contexts

        unique_nouns = list(set(all_nouns))
        self.analyze(unique_nouns)
        # We need to keep track of the contexts that were extracted for the analyze function
        self.extracted_contexts = ec.ex
        return unique_nouns

    def analyze(self, unique_nouns):
        ex = self.extracted_contexts
        all_docs = self.pdo.all_docs

        if ex:
            print("Performed %s total extractions" % len(ex.extractions))
            print("We have extracted %s new unique nouns" % len(unique_nouns))

            documents_with_extractions = list(ex.extractions.keys())
            print("Extracted from %s total documents" % len(documents_with_extractions))
            coverage = len(documents_with_extractions) / len(all_docs)
            print("Extraction coverage: %.4f percent" % coverage)
        else:
            print("No contexts have been extracted")
