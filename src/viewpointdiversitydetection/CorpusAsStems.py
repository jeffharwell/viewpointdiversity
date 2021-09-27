from nltk import SnowballStemmer


class CorpusAsStems:
    """
    This is a generalized class. You initialize it with the filter you want to use and it
    will process a list of Spacy documents, extract the stems which match your filter,
    and create a set of data structures that are useful for further analysis.

    :param token_filter: the function we will use to filter our tokens before stemming
                         it should take a Spacy token object and return either a boolean
                         which indicates whether it is a valid stem.
    """

    def __init__(self, token_filter):
        """
        Data structure that hold each stem in the corpus
        and the number of documents that it appears in.

        :param token_filter:
        """

        # Every document in the corpus gets an index number, this hold them
        self.all_doc_indexes = []
        # Hold a count documents that contain each stem {'stem1':doc count, 'stem2':doc count, ...}; this is each
        # stems document frequency
        self.stem_document_count = {}
        # Holds the the list of documents that contain each stem {'stem1':[doc idx1, doc idx2 ...], 'stem2':[doc
        # idx4, ...], ...}
        self.stem_to_doc = {}
        # Holds the list of stems contains in each documents {'doc idx1':[stem1, stem2, stem3 ...], 'doc idx2':[
        # stem4, ...], ...}
        self.doc_to_stem = {}

        # Our token filter, determines what is in or out
        self.token_filter = token_filter

        # Our Stemmer
        self.stemmer = SnowballStemmer(language='english')

    def print_stats(self):
        """
        Print stats on the stems and indexes.

        :return: None
        """
        print("Indexed %s documents" % len(self.all_doc_indexes))
        print("Extracted %s stems" % len(self.stem_to_doc))

        number_of_docs = len(self.all_doc_indexes)
        stem_document_counts_percentages = {}
        for stem, count in self.stem_document_count.items():
            stem_document_counts_percentages[stem] = int(count) / number_of_docs

        # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
        # I don't really understand the syntax here
        stem_sorted_by_coverage = {k: v for k, v in
                                   sorted(stem_document_counts_percentages.items(), key=lambda item: item[1],
                                          reverse=True)}
        most_common = [k for k in stem_sorted_by_coverage.keys()][0:5]
        print("Five most common stems by percent of documents in the corpus which contain them:")
        for s in most_common:
            print("  %s, coverage %.4f" % (s, stem_sorted_by_coverage[s]))

    def extract_stems(self, all_docs):
        """
        Extract all the stems from a list of Spacy documents and index them.

        :param all_docs:  all_docs is a list of Spacy document objects
        :return: None
        """
        for i, doc in enumerate(all_docs):
            self.all_doc_indexes.append(i)
            unique_stems = []
            self.doc_to_stem[i] = []
            for token in doc:
                # We are going to ignore stop words, punctuation, and spacing
                # if not token.is_space and not token.is_punct and not token.is_stop:
                if self.token_filter(token):
                    stem = self.stemmer.stem(token.text)
                    if stem not in unique_stems and stem != '':
                        unique_stems.append(stem)
            for stem in unique_stems:
                # First we keep a dictionary of the doc count per stem
                if stem in self.stem_document_count:
                    self.stem_document_count[stem] = self.stem_document_count[stem] + 1
                else:
                    self.stem_document_count[stem] = 1

                # Now we build an index of stem to doc
                if stem not in self.stem_to_doc:
                    self.stem_to_doc[stem] = [i]
                else:
                    self.stem_to_doc[stem].append(i)

                # And an index for doc to stem
                self.doc_to_stem[i].append(stem)
