import hashlib

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

    def generate_feature_vectors_from_sentences(self, extract_contexts, context_label, parsed_documents_object):
        """
        Generate the feature vectors for the document using the full sentence from the processed documents
        as opposed to just using the pre-processed, stopword filtered, tokens from the context.

        :param extract_contexts: dictionary of extracted TermContext objects by context label and docid
        :param context_label: the label used to identify the set of extracted contexts
        :param parsed_documents_object: a ParsedDocuments Object (usually ParsedDocumentsFourForums)
        :return: dictionary with document id as the key and an average SBERT embedding representing the context
        """
        #print(f"Running generate_feature_vectors_from_sentences for context '{context_label}'")
        # There are some contexts, create our vector

        # extract_contexts dictionary is of the form
        # {context_label: {doc_id: [TermContext1, TermContext2, ..], ..}, context_label2: ...}

        doc_idx_to_vector = {}
        # Iterate through every document that has an extraction
        for doc_idx, term_contexts in extract_contexts[context_label].items():
            vectors = []
            sentences_by_hash = {}
            parsed_document = parsed_documents_object.all_docs[doc_idx]

            # for every extracted context grab the sentences that correspond to
            # the sentence indices
            for term_context in term_contexts:
                for sentence_indices in term_context.sentence_indices:
                    sentences = self.get_sentences_from_indicies(sentence_indices)
                    for s in sentences:
                        # We only want unique sentences. Because we are moving from contexts to sentences
                        # it is possible that two contexts contain part of the same sentence, and we don't want
                        # to double count a single sentence in the embedding
                        sentence_hash = hashlib.md5(s.text.encode('utf-8')).hexdigest()
                        if sentence_hash not in sentences_by_hash:
                            # this one is new, add it, keyed by hash, to our dict of sentences
                            sentences_by_hash[sentence_hash] = s.text
                            # print(s.text)

            # Get the Sentence Bert Embedding for each sentence and add it to our list of vectors
            vectors.append(self._get_vector_from_term_sentences(sentences_by_hash.values()))

            # Check for zeros
            if len(vectors) == 0:
                # no contexts were extracted, use the zero vector
                vector = np.zeros(self.vector_size)
            else:
                vector = self._average_vector(vectors)

            # Add the average vector to our vector by document dictionary
            doc_idx_to_vector[doc_idx] = vector

        return doc_idx_to_vector

    def generate_feature_vectors(self, contexts_by_docid):
        """
        Function which sweeps through our dictionary and contexts per document
        and create a single BERT embedding vector representing those contexts.

        :param contexts_by_docid: dictionary with document indexes as keys and lists of TermContext objects as values
        :return: dictionary with document id as the key and an average SBERT embedding representing the context
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

    def get_sentences_from_indicies(self, sentence_indices, doc):
        """
        Takes a list of sentence indicies and returns a list of Spacy sentence objects

        :param sentence_indices: a list of integers representing sentences in the document
        :param doc: the document as a Spacy object
        :return: a list of sentences as Spacy spans
        """
        text = list()
        # doc.sents is a generator, so we can't just snag the sentence by index, we have to iterate
        for i, sentence in enumerate(doc.sents):
            if i in sentence_indices:
                text.append(sentence)
        return text

    def _get_vector_from_term_sentences(self, sentences):
        """
        Get the vector representation of each sentence in our context. Average the vectors,
        and then return a single vector representing this context.

        :param sentences: a list of sentences as strings
        """
        vectors = []

        if len(sentences) == 0:
            print("Warning: called getVector on an empty set of sentences. Returning zeros vector")
            vectors.append(np.zeros(self.vector_size))

        # Get all the tokens from our contexts, put them into space delimited string
        # then pass that to Bart and pull the value of the pooled_output to get the
        # embedding
        for sentence in sentences:
            vectors.append(self.bert_model.encode(sentence))

        # Check for undefined values (shouldn't be any for BERT, I think)
        for v in vectors:
            if not np.isfinite(v).all():
                print("In SBertFeatureGenerator the following vector has undefined values")
                print(v)

        # Average the context vectors to get a single BERT embedding for
        # the document.
        sum_of_all_vectors = np.sum(vectors, axis=0)  # note axis, otherwise np.sum collapses everything
        avg_of_all_vectors = np.divide(sum_of_all_vectors, len(vectors))

        return avg_of_all_vectors
