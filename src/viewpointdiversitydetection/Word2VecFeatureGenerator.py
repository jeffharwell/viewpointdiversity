import numpy as np
import gensim.downloader as api


class Word2VecFeatureGenerator:
    def __init__(self, vector_model=False):
        if not vector_model:
            print("Leading Gensim Model")
            # Load a small, fast, gensim word2vec pretrained model
            self.vector_model = api.load('glove-twitter-25')
            print("Finished")
        else:
            self.vector_model = vector_model

        self.vector_size = self.vector_model.vector_size
        self.include_zeros_in_averages = True

    def include_zeros_in_averages(self):
        """
        Instruct the object to not include zero vectors in the vector average when creating a feature vector.
        """
        self.include_zeros_in_averages = True

    def exclude_zeros_from_averages(self):
        """
        Instruct the object to include zero vectors in the average when creating feature vectors.
        """
        self.include_zeros_in_averages = False

    def _in_w2v_vocab(self, word):
        """
        Return true if the word is in the model vocabulary, false otherwise.

        :param word: string with the word we are to check
        """
        try:
            self.vector_model.key_to_index[word]
            return True
        except KeyError:
            return False

    def _get_vec(self, word):
        """
        Gets the normalized vector representation of a word form the model. If the word
        is not in the model's vocabulary it returns an empty element vector.

        :param word: string with the word to convert a vector.
        """
        if self._in_w2v_vocab(word):
            return self.vector_model.get_vector(word, norm=True)
        else:
            return np.zeros(self.vector_size)

    def _average_vector(self, vectors):
        """
        Takes a list of vectors and returns the average.

        :param vectors: a list of numpy vectors
        """
        for v in vectors:
            if not np.isfinite(v).all():
                print("Vector has undefined values")
                print(v)
        # Use numpy to create an average of all vectors
        sum_of_all_vectors = np.sum(vectors, axis=0)  # note axis, otherwise np.sum collapses everything
        if self.include_zeros_in_averages:
            avg_of_all_vectors = np.divide(sum_of_all_vectors, len(vectors))
        else:
            # We are not including zero vectors in the average calculation, so compute the number of
            # non-zero vectors we have and then use that number in the denominator when computing the average
            zero_vector = np.zeros(self.vector_size)
            length_nonzero_vectors = sum([1 for i in vectors if not np.array_equal(i, zero_vector)])
            if length_nonzero_vectors == 0:
                # all features are 0, return the sum as the avg, since both are 0
                avg_of_all_vectors = sum_of_all_vectors
            else:
                avg_of_all_vectors = np.divide(sum_of_all_vectors, length_nonzero_vectors)

        return avg_of_all_vectors

    def generate_feature_vectors(self, contexts_by_docid):
        """
        Function which sweeps through our dictionary and contexts per document
        and create a single word2vec vector representing those contexts.

        :param contexts_by_docid: dictionary with document indexes as keys and lists of TermContext objects as values
        """

        # There are some contexts, create our vector
        doc_idx_to_vector = {}
        for doc_idx in contexts_by_docid.keys():
            vectors = []
            for tc in contexts_by_docid[doc_idx]:
                vectors.append(self._get_vector_from_term_context(tc))
            if len(vectors) == 0:
                # no contexts were extracted, use the zero vector
                vector = np.zeros(self.vector_size)
            else:
                vector = self._average_vector(vectors)
            doc_idx_to_vector[doc_idx] = vector

        return doc_idx_to_vector

    def _get_vector_from_term_context(self, term_context_obj):
        """
        Get the vector representation of each token in our context. Average the vectors,
        and then return a single vector representing this context.

        :param term_context_obj: A Gensim vector model
        """
        if not term_context_obj.has_context():
            print("Warning: called getVector on an empty TermContext object")

        # Generate a flat list that contains a vector for every token in our context
        # note if the word is not in the vector models vocabulary we replace it with a
        # vector of zeros, np.zeros(self.vector_size).
        vectors = []
        for context_structure in term_context_obj.contexts:
            leading_vectors = [self._get_vec(w) for w in context_structure['leading_tokens']]
            trailing_vectors = [self._get_vec(w) for w in context_structure['trailing_tokens']]
            vectors = vectors + leading_vectors + trailing_vectors

        for v in vectors:
            if not np.isfinite(v).all():
                print("In TermContext the following vector has undefined values")
                print(v)

        # Create
        sum_of_all_vectors = np.sum(vectors, axis=0)  # note axis, otherwise np.sum collapses everything
        avg_of_all_vectors = np.divide(sum_of_all_vectors, len(vectors))
        return avg_of_all_vectors
