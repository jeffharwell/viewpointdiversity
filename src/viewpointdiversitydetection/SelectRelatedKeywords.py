import math
import statistics
import string

import pandas
from nltk.corpus import stopwords

from viewpointdiversitydetection import ExtractContexts, NounTokenFilter


class SelectRelatedKeywords:
    """
    The ExtractRelatedNouns class supports various approaches to extracting lists of related
    nouns from a corpus given a set of search terms.
    """
    def __init__(self, pdo, doc_indices, search_terms, stemmer):
        self.pdo = pdo
        self.doc_indices = doc_indices
        self.search_terms = search_terms
        self.search_stems = [stemmer.stem(s) for s in search_terms]
        self.coverage_by_k = {}  # data structure to store all of our coverages by k value for later analysis
        self.co_occurrence_threshold = 1  # number of times a term has to co-occur before it is considered
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = stemmer

        # Create new indices that map stems to document indices and document indices to stems
        # but only contain the documents that are in doc_indicies (presumable our training set)
        self.doc_to_stem = {doc_idx: stems for (doc_idx, stems) in pdo.cas.doc_to_stem.items() if
                            doc_idx in doc_indices}
        self.stem_to_doc = {}
        for stem, doc_list in pdo.cas.stem_to_doc.items():
            filtered_doc_list = [i for i in doc_list if i in self.doc_indices]
            if len(filtered_doc_list) > 0:
                self.stem_to_doc[stem] = filtered_doc_list

        # Extract all related nouns as per Spacy
        # All contexts will hold a list of vdd.TermContext objects
        # indexed by document index
        self.all_contexts = self.extract_all_contexts(pdo)
        if len(self.all_contexts) == 0:
            raise ValueError(f"No contexts were extracted from corpus for search terms {self.search_terms}")

        # Prove that the tokens we are getting from the extracted contexts
        # have been filtered correctly (hint, they haven't)
        self.verify_tokens_in_contexts_with_stem_index()

        # Generate a dataframe that has the related term in the term column
        # and the co-occurrence weight in the co_weight column, it in sorted
        # by co_weight descending.
        self.coterm_df = self.generate_cooccurrence_structure()

        # Stem each term and add the IDF by stem
        self.coterm_df['stem'] = self.coterm_df.apply(lambda x: stemmer.stem(x['term']), axis='columns')
        self.coterm_df['idf'] = self.coterm_df.apply(lambda x: self.calculate_idf(x['stem']), axis='columns')

    def extract_all_contexts(self, pdo, context_size=4):
        """
        Extract all of related nouns from a corpus.

        :param pdo: the parsed document object
        :param context_size: the range of tokens to extract for related nouns, defaults to 4
        :return: all extracted contexts
        """
        all_contexts = {}
        ec = ExtractContexts(pdo, context_size, NounTokenFilter())
        for doc_idx in self.doc_indices:
            doc = pdo.all_docs[doc_idx]
            extracted_contexts = ec.get_contexts_for_multiple_terms(doc, doc_idx, self.search_terms, 'related')
            if len(extracted_contexts) > 0:
                all_contexts[doc_idx] = extracted_contexts
        # print(f"Extracted related nouns from {len(all_contexts) / len(self.doc_indices):.2f} percent of documents")
        return all_contexts

    def verify_tokens_in_contexts_with_stem_index(self):
        for doc_idx, c in self.all_contexts.items():  # iterate through all the extracted contexts
            if len(c) > 0:  # some documents had no contexts extracted, skip these
                for tc in c:  # for each term context in the extracted contexts (should only be one)
                    for mc in tc.contexts:  # for each matching context
                        all_tokens = [t.lower() for t in mc['leading_tokens'] + mc[
                            'trailing_tokens']]  # get all the terms from the context
                        for t in all_tokens:
                            #
                            # This is a bit crazy
                            # Looking at the logic from ExtractContexts it seems that every token needs to be
                            # passed through the token filter before it can be collected, so I don't understand
                            # why we are seeing stopwords a punctuation in the collected tokens!!
                            if self.our_monkey_patch_token_filter(t) and \
                                    self.stemmer.stem(t) not in self.pdo.cas.stem_to_doc:
                                print(
                                    f"Couldn't find stem {self.stemmer.stem(t)} "
                                    f"from term {t} from extracted context in pdo.cas.stem_to_doc index")

    def our_monkey_patch_token_filter(self, token_string):
        """
        This is a monkey patch, we really shouldn't have to do this kind of filter of the extracted tokens
        They should already have passed through a token filter.
        We need to debug the ExtractContexts.py logic for the documents where we have junk context tokens
        and figure out what is going wrong!!
        """
        t = token_string
        s = self.stemmer.stem(t)
        if s in self.pdo.cas.stem_to_doc and t.lower() not in self.stop_words\
                and t not in string.punctuation\
                and len(t) > 1:
            # if t.lower() not in self.stop_words and t not in string.punctuation and len(t) > 1:
            # Passes the filter
            return True
        else:
            # It doesn't pass the filter
            return False

    def generate_cooccurrence_structure(self):
        # Create a data structure with number of times each
        # term in the extracted contexts occurs with each other
        # term in the extracted contexts. This is a context co-occurrence
        # matrix of sorts.
        tco = {}  # term co-occurrence
        for doc_idx, c in self.all_contexts.items():  # iterate through all the extracted contexts
            if len(c) > 0:  # some documents had no contexts extracted, skip these
                for tc in c:  # for each term context in the extracted contexts (should only be one)
                    for mc in tc.contexts:  # for each matching context
                        all_tokens = [t.lower() for t in mc['leading_tokens'] + mc['trailing_tokens'] if
                                      self.our_monkey_patch_token_filter(t)]  # get all the terms from the context
                        for t in all_tokens:  # for every token t
                            if t not in tco:  # if we have not see it before create an empty structure
                                tco[t] = {}
                            for ot in [ot for ot in all_tokens if ot != t]:  # for every other token in the context
                                if ot in tco[t]:  # if we have seen this co-occurrance before then increment the count
                                    tco[t][ot] += 1
                                else:  # this is the first time we have seen this combination, it has co-occurred once
                                    tco[t][ot] = 1

        # Now filter the co-occurrence matrix
        # if the two terms don't appear together more than twice, or the token
        # itself is less than 2 characters long, then discard it
        ftco = {}
        co_occurence_threshold = self.co_occurrence_threshold
        for term, co in tco.items():  # term is our term, co is our dictionary of co-occurrence terms with counts
            if len(term) > 2 and len(co.values()) > 0 and max(
                    co.values()) >= co_occurence_threshold:  # make sure we have co-occurrences over our threshold
                ftco[term] = {}  # make the data structure
                for coterm, count in co.items():  # go through each co-occurrence item
                    if count >= co_occurence_threshold and len(coterm) > 2:  # if it meets the filter criteria
                        ftco[term][coterm] = count  # add it to the new structure

        # For each term we indicate the number of
        # other terms that it co-occurs with, relative
        # to the number of terms extracted. A relative
        # measure of the centrality of a related term,
        # normalized by the number of terms extracted
        num_co_occ = {}
        for k, v in ftco.items():
            try:
                num_co_occ[k] = len(v) / len(ftco)
            except ValueError as e:
                print(f"Warning, get error {e} when normalizing the co-occurrence value.")
                print(len(v))
                print(len(tco))

        frame_data = []
        for t, c in num_co_occ.items():
            frame_data.append({'term': t, 'co_weight': c})

        if len(frame_data) == 0:
            raise RuntimeError('No co-occurrence weights could be created from corpus')

        co_term_df = pandas.DataFrame(frame_data)
        co_term_df.sort_values('co_weight', ascending=False, inplace=True)

        return co_term_df

    def calculate_doc_frequency(self, stem):
        if stem in self.stem_to_doc:
            return len(self.stem_to_doc[stem])
        else:
            return 0

    def calculate_idf(self, stem):
        doc_freq = self.calculate_doc_frequency(stem)
        if doc_freq == 0:
            return 0
        else:
            num_docs = len(self.doc_indices)
            return math.log2(num_docs / doc_freq)

    def calculate_doc_coverage(self, stems):
        num_docs = len(self.doc_indices)
        docs_covered = []
        for s in stems:
            if s in self.stem_to_doc:
                docs_covered += self.stem_to_doc[s]
        num_docs_covered = len(set(docs_covered))
        return num_docs_covered / num_docs

    def num_docs_covered(self, stems):
        docs_covered = []
        for s in stems:
            if s in self.stem_to_doc:
                docs_covered += self.stem_to_doc[s]
        num_docs_covered = len(set(docs_covered))
        return num_docs_covered

    def mean_extracted_contexts(self, ec_obj, stem_list, sorted_terms_df):
        """
        Do it the hard way. Actually do the extraction for the list of terms and report on the
        number of contexts extracted.

        Note that we are really counting the number of different terms that extract contexts, a term
        might actually generate multiple contexts for a document, but we only count it once.
        This is because we are looking for multiple concepts, because viewpoint is multiple concepts in
        relationship.
        """
        terms_from_stems = []
        for s in stem_list:
            terms_from_stems += sorted_terms_df[sorted_terms_df['stem'] == s]['term'].tolist()
        # print(f"    Stems: {len(stem_list)} Terms: {len(terms_from_stems)}")

        ec_obj.extract_additional_contexts({'search': self.search_terms, 'testing context': terms_from_stems})

        num_extracts_by_doc_idx = {}
        for i in range(0, len(self.pdo.all_docs)):
            num_extracts_by_doc_idx[i] = 0

        for label in ['search', 'testing context']:
            for doc_idx, term_contexts in ec_obj.contexts[label].items():  # go through each extracted context
                if len(term_contexts) > 0:  # if any contexts were extracted
                    # count them, note that a term that generates multiple contexts is
                    # still counted a one extraction
                    if doc_idx in num_extracts_by_doc_idx:
                        num_extracts_by_doc_idx[doc_idx] += len(term_contexts)
                    else:
                        num_extracts_by_doc_idx[doc_idx] = len(term_contexts)

        mean_contexts_extracted = statistics.mean(list(num_extracts_by_doc_idx.values()))
        # print(f"    Mean Contexts Extracted: {mean_contexts_extracted}")
        return mean_contexts_extracted

    def find_terms_from_sorted_df_by_context(self, sorted_terms_df, threshold, context_size):
        """
        Select additional terms until the average number of extracted contexts is 'threshold' across the corpus
        """
        # Create the list of stems from the list of terms
        distinct_ordered_stems = []
        for t in sorted_terms_df['term'].to_list():
            s = self.stemmer.stem(t)
            if s not in distinct_ordered_stems:
                distinct_ordered_stems.append(s)

        # use the ExtractContext object to keep track of, and calculate, our mean contexts extracted
        ec = ExtractContexts(self.pdo,
                             context_size)
        cumulative_stems = []  # holds our list of selected stems
        for s in distinct_ordered_stems:  # iterate through our list of stems
            cumulative_stems.append(s)  # these are all the stems we have tried
            if self.mean_extracted_contexts(ec, cumulative_stems,
                                            sorted_terms_df) >= threshold:  # calculate the mean contexts extracted
                # we are done, mean extracted contexts is > 1
                break

        terms_from_stems = []
        for s in cumulative_stems:
            terms_from_stems += sorted_terms_df[sorted_terms_df['stem'] == s]['term'].tolist()
        # print(f"{len(cumulative_stems)} top related stems generated {len(terms_from_stems)} terms.")
        # print(f"   First Stem: {cumulative_stems[0]}, First Term: {terms_from_stems[0]}")
        return terms_from_stems

    def find_terms_from_sorted_df(self, sorted_terms_df, k=None):
        """
        Returns a list of terms in order of weight descending.

        :param sorted_terms_df: a dataframe with a 'term' column sorted in the desired order
        :param k: the 'k' value, can be empty if 'k' is not being used
        """

        # Calculate cumulative coverage per additional stem added in order of weight descending
        cumulative_stems = []
        stem_and_coverage = []
        idx = 0
        for s in [self.stemmer.stem(t) for t in sorted_terms_df['term'].to_list()]:  # list of weighted stems in order
            if s not in self.search_stems:  # don't want to include a stem that is already in the search terms
                cumulative_stems.append(s)
                stem_and_coverage.append(
                    {'stem': s, 'coverage': self.calculate_doc_coverage(cumulative_stems), 'idx': idx})
                idx += 1
        # print(stem_and_coverage[:10])
        if k:
            self.coverage_by_k[k] = stem_and_coverage

        # Now find the top X related terms based on the coverage and threshold values
        top_related_stems = []

        previous_docs_covered = 0
        selected_stems = []
        for s_and_c in stem_and_coverage:
            selected_stems.append(s_and_c['stem'])
            delta = self.num_docs_covered(selected_stems) - previous_docs_covered
            if delta < 1:
                """
                print(f"last s_and_c evaluated: {s_and_c}")
                print(
                    f"last selected stem {s_and_c['stem']} "
                    f"and docs covered: {self.num_docs_covered([s_and_c['stem']])}")
                print(
                    f"Done. Completed with stem index '{s_and_c['stem']}' {s_and_c['idx']} with cumulative "
                    f"coverage {s_and_c['coverage']}")
                """
                break
            else:
                previous_docs_covered = self.num_docs_covered(selected_stems)
                top_related_stems.append(s_and_c['stem'])

        terms_from_stems = []
        for s in top_related_stems:
            terms_from_stems += sorted_terms_df[sorted_terms_df['stem'] == s]['term'].tolist()
        # print(f"{len(top_related_stems)} top related stems generated {len(terms_from_stems)} terms.")
        # print(f"   First Stem: {top_related_stems[0]}, First Term: {terms_from_stems[0]}")
        return terms_from_stems

    def get_related_keywords_idf(self, idf_method):
        """
        Get the related keywords using just IDF, sorted either ascending or descending

        :param idf_method: string, either 'asc' or desc'
        """

        if idf_method == 'asc':
            ascending = True
        elif idf_method == 'desc':
            ascending = False
        else:
            raise RuntimeError(f"idf_method must be 'asc' or 'desc', not {idf_method}")

        # Sort by idf score, either ascending or descending
        sorted_terms_df = self.coterm_df.sort_values('idf', ascending=ascending, inplace=False)

        return self.find_terms_from_sorted_df(sorted_terms_df)

    def get_related_keywords(self, k):
        """
        :param k: weight to give the IDF score
        """

        # Use the k value to calculate our weights
        self.coterm_df['weighted'] = self.coterm_df.apply(lambda x: math.log2(k * x['co_weight']) * x['idf'],
                                                          axis='columns')

        # Sort by weight descending
        sorted_terms_df = self.coterm_df.sort_values('weighted', ascending=False, inplace=False)

        return self.find_terms_from_sorted_df(sorted_terms_df, k)

    def get_related_keywords_context_threshold(self, k, context_size, mean_num_contexts_extracted=2):
        """
        Weighs co-terms with 'k' and then selects terms until the average number of concepts
        extracted from the terms/stems is 2 including the search terms.

        :param k: weight to give the IDF score
        :param context_size: the context size to use when extracting contexts
        :param mean_num_contexts_extracted: the mean number of contexts extracted used to determine when to stop
               selecting new terms from our list of sorted candidate terms. Defaults to 2.
        """

        # Use the k value to calculate our weights
        self.coterm_df['weighted'] = self.coterm_df.apply(lambda x: math.log2(k * x['co_weight']) * x['idf'],
                                                          axis='columns')

        # Sort by weight descending
        sorted_terms_df = self.coterm_df.sort_values('weighted', ascending=False, inplace=False)

        return self.find_terms_from_sorted_df_by_context(sorted_terms_df, mean_num_contexts_extracted, context_size)

    def get_related_keywords_context_threshold_idf(self, idf_method, context_size, mean_num_contexts_extracted=2):
        """
        Get the related keywords using just IDF and context threshold, sorted either ascending or descending

        :param idf_method: string, either 'asc' or desc'
        :param context_size: the context size to use when extracting contexts
        :param mean_num_contexts_extracted: the mean number of contexts extracted used to determine when to stop
               selecting new terms from our list of sorted candidate terms. Defaults to 2.
        """

        if idf_method == 'asc':
            ascending = True
        elif idf_method == 'desc':
            ascending = False
        else:
            raise RuntimeError(f"idf_method must be 'asc' or 'desc', not {idf_method}")

        # Sort by idf score, either ascending or descending
        sorted_terms_df = self.coterm_df.sort_values('idf', ascending=ascending, inplace=False)

        return self.find_terms_from_sorted_df_by_context(sorted_terms_df, mean_num_contexts_extracted, context_size)
