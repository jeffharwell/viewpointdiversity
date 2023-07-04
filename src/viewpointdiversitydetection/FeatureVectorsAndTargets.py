from viewpointdiversitydetection.Word2VecFeatureGenerator import Word2VecFeatureGenerator
from viewpointdiversitydetection.ExtractContexts import ExtractContexts
from viewpointdiversitydetection.SentimentFeatureGenerator import SentimentFeatureGenerator
from viewpointdiversitydetection.feature_vector_creation_utilities import *


class FeatureVectorsAndTargets:
    """
    Create a set of feature vectors and associated target classes from a ParsedDocumentsForForums object.
    """

    def __init__(self, parsed_docs, vector_model, search_terms, related_terms, context_size):
        """
        Initialize the object

        :param parsed_docs: a ParsedDocumentsFourForums object
        :param vector_model: a Gensim Word2Vec model
        :param search_terms: a list of search terms
        :param related_terms: a list of terms related to the search terms by context
        :param context_size: integer containing the number of terms to consider as context when creating features
        """
        self.pdo = parsed_docs
        self.vector_model = vector_model
        self.search_terms = search_terms
        self.related_terms = related_terms
        self.context_size = context_size
        self.contexts = []  # contexts indexed by term to extract then the document index number
        # {context_label: {doc_id: [TermContext1, TermContext2, ..], ..}, context_label2: ...}

        self.feature_vectors = {}  # indexed by document index from the PDO
        self.feature_vectors_as_components = {}  # indexed by document index from the PDO
        self.targets_for_features = {}  # indexed by document index from the PDO

        self.length_of_sentiment_vector = 0
        self.length_of_word2vec_vector = 0
        self.empty_sentiment_vector = [0.0]
        self.empty_word2vec_vector = [0.0]

        self.pass_sentences_for_feature_extraction = False

    def create_feature_vectors_and_targets(self, verbose_level=1):
        """
        Instructs the object to create and store the feature vectors and their associated target classes. The
        results are stored in the attributes 'feature_vectors' and 'targets_for_features'.

        Verbosity levels of greater than 1 will print out the indices of the documents that do not have any
        contexts extracted. Verbosity level of 1 gives that information in summary form.

        :param verbose_level: the verbosity of the output, currently 0, 1, or greater than 1
        """

        #
        # Extract Contexts
        #
        terms_for_extraction = {'search': self.search_terms, 'related': self.related_terms}
        ec = ExtractContexts(self.pdo, self.context_size)
        ec.extract_contexts(terms_for_extraction)
        self.contexts = ec.contexts  # save the details for the contexts we extracted
        number_of_search_contexts = sum([len(c) for c in ec.get_contexts_by_doc_id_for('search').values()])
        number_of_related_contexts = sum([len(c) for c in ec.get_contexts_by_doc_id_for('related').values()])
        print(f"{number_of_search_contexts} search contexts extracted")
        print(f"{number_of_related_contexts} related contexts extracted")

        #
        # Create the Sentiment Vectors
        #
        avg_measures = ['compound', 'polarity', 'subjectivity']
        max_measures = ['compound', 'polarity', 'subjectivity']
        min_measures = ['compound', 'polarity', 'subjectivity']
        sfg_obj = SentimentFeatureGenerator(avg_measures, max_measures, min_measures, False)
        sfg_obj.exclude_zeros_from_averages()
        search_sentiment_vectors_by_doc_id = sfg_obj.generate_feature_vectors(ec.get_contexts_by_doc_id_for('search'))
        related_sentiment_vectors_by_doc_id = sfg_obj.generate_feature_vectors(ec.get_contexts_by_doc_id_for('related'))

        #
        # Create the Word2Vec Features
        #
        def get_w2v_obj(vm):
            # Figure out which embedding generator to use by the name of the object that got
            # passed as the vector_model.
            if vm.__class__.__name__ == 'KeyedVectors':
                # This is a Gensim vector model, create the w2v_obj
                # using the Word2VecFeatureGenerator
                w2v_obj = Word2VecFeatureGenerator(self.vector_model)
                return w2v_obj
            elif vm.__class__.__name__ == 'BertFeatureGenerator':
                # We've been passed a BertFeatureGenerator, it has the
                # same interface as a Word2VecFeatureGenerator, so just us it
                return vm
            elif vm.__class__.__name__ == 'SBertFeatureGenerator':
                # We've been passed an SBertFeatureGenerator object, it has the
                # same interface as a Word2VecFeatureGenerator
                self.pass_sentences_for_feature_extraction = True  # It wants the full sentences
                return vm
            else:
                RuntimeError(f"We don't know how to handle a {vm.__class__.__name__}. "
                             f"Either pass a Gensim vector model or a BertFeatureGenerator object")

        w2v_obj = get_w2v_obj(self.vector_model)
        w2v_obj.exclude_zeros_from_averages()  # don't include zero vectors when computing the vector averages
        if self.pass_sentences_for_feature_extraction:
            # We will pass the contexts pull the parsed document object so that the embedding generator
            # has access to the full sentence, not just the extracted context tokens
            search_word2vec_vectors_by_doc_id = w2v_obj.generate_feature_vectors_from_sentences(self.contexts,
                                                                                                'search', self.pdo)
            related_word2vec_vectors_by_doc_id = w2v_obj.generate_feature_vectors_from_sentences(self.contexts,
                                                                                                 'related', self.pdo)
        else:
            # The embedding generator just needs to extracted context tokens
            search_word2vec_vectors_by_doc_id = w2v_obj.generate_feature_vectors(
                ec.get_contexts_by_doc_id_for('search'))
            related_word2vec_vectors_by_doc_id = w2v_obj.generate_feature_vectors(
                ec.get_contexts_by_doc_id_for('related'))

        # We have to figure out the size of our feature vectors so that we can create empty
        # vectors of the right size.
        first_id = list(search_sentiment_vectors_by_doc_id.keys())[0]
        self.length_of_sentiment_vector = len(search_sentiment_vectors_by_doc_id[first_id])
        first_id = list(search_word2vec_vectors_by_doc_id.keys())[0]
        self.length_of_word2vec_vector = len(search_word2vec_vectors_by_doc_id[first_id])
        self.empty_sentiment_vector = [0.0] * self.length_of_sentiment_vector
        self.empty_word2vec_vector = [0.0] * self.length_of_word2vec_vector

        # We may have a certain number of documents that did not have any contexts extracted
        # We don't want to include those in our final feature set.
        doc_idx_with_no_extractions = []
        all_labels = self.contexts.keys()
        for i in range(0, len(self.pdo.all_docs)):  # for every parsed document
            has_context = [1 for l in all_labels if i in self.contexts[l]]  # does this document have any contexts
            if sum(has_context) == 0:  # we don't find any extracted contexts for any label
                doc_idx_with_no_extractions.append(i)  # i is the index of the document that had no context extracted

        # Warn the programmer that we have a certain number of documents that had no context and will not be included
        # in the feature set.
        if len(doc_idx_with_no_extractions) > 0:
            if verbose_level > 0:
                print(f"INFO: There were {len(doc_idx_with_no_extractions)} documents where no context was extracted:")
                print("      The following documents will not be included in the feature set.")
                percent_of_corpus = len(doc_idx_with_no_extractions)*100.0/len(self.pdo.all_docs)
                print(f"      This represents {percent_of_corpus:.2f}% of the corpus")
            if verbose_level > 1:
                for i in doc_idx_with_no_extractions:
                    print(i, end=", ")
        print("")


        #
        # Combine the Word2Vec Features and the Sentiment Features into a single feature vector
        #
        for i in range(0, len(self.pdo.all_docs)):
            if i not in doc_idx_with_no_extractions:  # if it doesn't have a context extraction don't include it
                if i in search_word2vec_vectors_by_doc_id:
                    search_word2vec = search_word2vec_vectors_by_doc_id[i]
                else:
                    search_word2vec = self.empty_word2vec_vector

                if i in related_word2vec_vectors_by_doc_id:
                    related_word2vec = related_word2vec_vectors_by_doc_id[i]
                else:
                    related_word2vec = self.empty_word2vec_vector

                if i in search_sentiment_vectors_by_doc_id:
                    search_sentiment = search_sentiment_vectors_by_doc_id[i]
                else:
                    search_sentiment = self.empty_sentiment_vector

                if i in related_sentiment_vectors_by_doc_id:
                    related_sentiment = related_sentiment_vectors_by_doc_id[i]
                else:
                    related_sentiment = self.empty_sentiment_vector

                sv = combine_as_average(search_sentiment, related_sentiment, include_zeros=True)
                self.feature_vectors[i] = sv + list(search_word2vec) + list(related_word2vec)
                self.feature_vectors_as_components[i] = {'sentiment': sv,
                                                         'search': search_word2vec,
                                                         'related': related_word2vec}
                self.targets_for_features[i] = self.pdo.target[i]


class Holder:
    """
    Holds just the features and targets, not the parsed documents themselves or the vector model.
    Useful for creating smaller pickle files.

    Attributes:
        feature_vectors                 a list of feature vectors, one entry per document
        feature_vectors_as_components   the feature vector but in a dictionary identifying the individual components
        targets_for_features            list containing the target for each feature
    """
    def __init__(self, database: str, topic_name: str, search_terms: list[str], stance_a: str, stance_b: str,
                 label_a: str, label_b: str, stance_agreement_cutoff: float, vector_model_short_name: str):
        """
        Initialized the object with critical meta-date about the features being stored.

        :param database: Name of the source database for the corpus the features were extracted from
        :type database: str
        :param topic_name: Name of the topic
        :type topic_name: str
        :param search_terms: list of strings contaning the search terms used to extract the context features
        :type search_terms: list[str]
        :param stance_a: stance a, the stance corresponding to target a
        :type search_terms: str
        :param stance_b: stance b, the stance corresponding to target b
        :type search_terms: str
        :param label_a: the target label for stance a
        :type label_a: str
        :param label_b: the target label for stance b
        :type label_b: str
        :param stance_agreement_cutoff: the inter-annotator agreement cutoff used when generating the features
        :type stance_agreement_cutoff: float
        :param vector_model_short_name: the short name of the vector model used to create the features
        :type vector_model_short_name: str
        """
        self.database = database
        self.topic_name = topic_name
        self.search_terms = search_terms.copy()
        self.stance_a = stance_a
        self.stance_b = stance_b
        self.label_a = label_a
        self.label_b = label_b
        self.stance_agreement_cutoff = stance_agreement_cutoff
        self.vector_model_short_name = vector_model_short_name
        self.feature_vectors = None
        self.feature_vectors_as_components = None
        self.targets_for_features = None

    def populate(self, fvt: FeatureVectorsAndTargets):
        """
        Copies the feature vectors, feature vectors as components, and targets
        from a FeatureVectorsAndTargets object to this Holder object.
        :param fvt: the FeatureVectorsAndTargets object you wish to copy
        :type fvt: FeatureVectorsAndTargets
        """
        self.feature_vectors = fvt.feature_vectors.copy()
        self.feature_vectors_as_components = fvt.feature_vectors_as_components.copy()
        self.targets_for_features = fvt.targets_for_features.copy()
