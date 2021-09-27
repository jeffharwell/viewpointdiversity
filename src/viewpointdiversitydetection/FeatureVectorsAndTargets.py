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
            self.targets_for_features.append(self.pdo.target[i])
