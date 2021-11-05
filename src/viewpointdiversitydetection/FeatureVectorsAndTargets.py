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

        self.feature_vectors = []
        self.feature_vectors_as_components = []
        self.targets_for_features = []

        self.length_of_sentiment_vector = 0
        self.length_of_word2vec_vector = 0
        self.empty_sentiment_vector = [0.0]
        self.empty_word2vec_vector = [0.0]

    def create_feature_vectors_and_targets(self):
        """
        Instructs the object to create and store the feature vectors and their associated target classes. The
        results are stored in the attributes 'feature_vectors' and 'targets_for_features'
        """

        #
        # Extract Contexts
        #
        terms_for_extraction = {'search': self.search_terms, 'related': self.related_terms}
        ec = ExtractContexts(self.pdo, self.context_size, terms_for_extraction)
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
        w2v_obj = Word2VecFeatureGenerator(self.vector_model)
        w2v_obj.exclude_zeros_from_averages()  # don't include zero vectors when computing the vector averages
        search_word2vec_vectors_by_doc_id = w2v_obj.generate_feature_vectors(ec.get_contexts_by_doc_id_for('search'))
        related_word2vec_vectors_by_doc_id = w2v_obj.generate_feature_vectors(ec.get_contexts_by_doc_id_for('related'))

        # We have to figure out the size of our feature vectors so that we can create empty
        # vectors of the right size.
        first_id = list(search_sentiment_vectors_by_doc_id.keys())[0]
        self.length_of_sentiment_vector = len(search_sentiment_vectors_by_doc_id[first_id])
        first_id = list(search_word2vec_vectors_by_doc_id.keys())[0]
        self.length_of_word2vec_vector = len(search_word2vec_vectors_by_doc_id[first_id])
        self.empty_sentiment_vector = [0.0] * self.length_of_sentiment_vector
        self.empty_word2vec_vector = [0.0] * self.length_of_word2vec_vector

        #
        # Combine the Word2Vec Features and the Sentiment Features into a single feature vector
        #
        for i in range(0, len(self.pdo.all_docs)):
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
            self.feature_vectors.append(sv + list(search_word2vec) + list(related_word2vec))
            self.feature_vectors_as_components.append({'sentiment': sv,
                                                       'search': search_word2vec,
                                                       'related': 'related_word2vec'})
            self.targets_for_features.append(self.pdo.target[i])


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
