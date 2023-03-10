import numpy as np
from sentence_transformers import SentenceTransformer

class SBertFeatureGenerator:
    def __init__(self, bert_model, output_shape):
        if not bert_model:
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = bert_model.encode('My Test Sentence')
            self.bert_model = bert_model
            self.vector_size = embedding.shape[0]
        else:
            self.bert_model = bert_model
            self.vector_size = output_shape

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

    def _average_vector(self, vectors):
        """
        Takes a list of vectors and returns the average. This explicitly handle the exclusion of
        zero vectors if indicated, which is what makes the logic worthy of an entire method.

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
        and create a single BERT embedding vector representing those contexts.

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

        # Get all the tokens from our contexts, put them into space delimited string
        # then pass that to Bart and pull the value of the pooled_output to get the
        # embedding
        vectors = []
        for context_structure in term_context_obj.contexts:
            # create the token string
            tokens = context_structure['leading_tokens'] + context_structure['trailing_tokens']
            token_string = " ".join(tokens)

            # Create the Embedding
            embedding = self.bert_model.encode(token_string)

            vectors.append(embedding)

        # Check for undefined values (shouldn't be any for BERT, I think)
        for v in vectors:
            if not np.isfinite(v).all():
                print("In TermContext the following vector has undefined values")
                print(v)

        # Average the context vectors to get a single BERT embedding for
        # the document.
        sum_of_all_vectors = np.sum(vectors, axis=0)  # note axis, otherwise np.sum collapses everything
        avg_of_all_vectors = np.divide(sum_of_all_vectors, len(vectors))

        return avg_of_all_vectors
