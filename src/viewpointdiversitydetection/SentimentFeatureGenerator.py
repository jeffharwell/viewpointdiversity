from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from viewpointdiversitydetection.TrackSentiment import TrackSentiment


class SentimentFeatureGenerator:
    """
    Generates a feature vector consisting for the measures specified for a
    set of extracted document contexts.
    """

    def __init__(self, avg_measures, max_measures, min_measures, include_search_term_in_calculation):
        """
        Initialize our SearchSentimentFeatureGenerator object.

        :param avg_measures: A list of sentiment keys for which we should calculate an average score per document
        :param max_measures: A list of sentiment keys for which we should calculate a maximum score per document
        :param min_measures: A list of sentiment keys for which we should calculate a minimum score per document
        :param include_search_term_in_calculation: Should the trigger term from the context be included in the
                                                   sentiment score calculation
        """
        self.avg_measures = avg_measures
        self.max_measures = max_measures
        self.min_measures = min_measures
        self.include_term = include_search_term_in_calculation
        self.include_zeros_in_averages = True

        number_of_features = len(avg_measures) + len(max_measures) + len(min_measures)
        self.empty_vector = [0.0] * number_of_features

        # Initialize the Vader Sentiment Analyzer
        self.vader = SentimentIntensityAnalyzer()

    def exclude_zeros_from_averages(self):
        """
        If this method is called the object will exclude zeros from any feature calculations of sentiment averages.
        """
        self.include_zeros_in_averages = False

    def include_zeros_in_averages(self):
        """
        If this method is called the object will include zero sentiment values in the calculation of sentiment averages.
        This is currently the default.
        """
        self.include_zeros_in_averages = True

    def generate_feature_vectors(self, contexts_by_docid):
        """
        Generate a list of feature vectors indexed by docid.

        :param contexts_by_docid: a dictionary structure that contains the contexts we are calculating
                                  sentiment features for indexed by the document id
        """
        vector_by_docid = {}
        for doc_idx, contexts in contexts_by_docid.items():
            vector_by_docid[doc_idx] = self.calculate_sentiment_from_contexts(contexts)

        return vector_by_docid

    def calculate_sentiment_from_contexts(self, contexts):
        """
        Calculate a sentiment vector. Note this calculates the sentiment vector
        for a single set of contexts from the same document. So it only returns one
        vector representing the avg, min, and max of the measures specified in
        the constructor.
        """
        sum_keys = self.avg_measures
        max_keys = self.max_measures
        min_keys = self.min_measures

        # Set up our TrackSentiment object
        # it does the most of the work, we just pass it
        # the sentiment scores for each context
        tracker = TrackSentiment(sum_keys, max_keys, min_keys, self.include_zeros_in_averages)

        # Get our vader object
        # vader = SentimentIntensityAnalyzer()
        for termcontext_obj in contexts:
            for tokens in termcontext_obj.contexts:
                if self.include_term:
                    all_tokens = tokens['leading_tokens'] + [termcontext_obj.term] + tokens['trailing_tokens']
                else:
                    all_tokens = tokens['leading_tokens'] + tokens['trailing_tokens']
                # Get sentiment score from Vader
                ss = self.vader.polarity_scores(" ".join(all_tokens))
                # Add in the score from TextBlob
                tb = TextBlob(" ".join(all_tokens))
                tbs = tb.sentiment
                ss['polarity'] = tbs.polarity
                ss['subjectivity'] = tbs.subjectivity
                tracker.process_sentiment(ss)
        if tracker.has_sentiment_vector():
            return tracker.get_sentiment_vector()
        else:
            return self.empty_vector
