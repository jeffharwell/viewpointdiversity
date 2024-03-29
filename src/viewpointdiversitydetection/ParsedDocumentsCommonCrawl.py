import spacy
from negspacy.termsets import termset
import torch

from viewpointdiversitydetection.TokenFilter import TokenFilter
from viewpointdiversitydetection.CorpusAsStems import CorpusAsStems

"""
# Class to Parse a Common Crawl dataset, it expect that you have already loaded the text into a dataframe
# from wherever it is being stored. Below is an example of lading the text from a MySQL database and then using
# the resulting dataframe to instantiate a ParsedDocumentsCommonCrawl document.

#
# Define the tokens we are interested in and then retrieve and pre-process our corpus
# This needs to be identical between the four fourums text pre-processing and the
# common crawl text pre-processing. So define it here.
tf = vdd.TokenFilter()

# Pull the data from the database
cc_con = pymysql.connect(host=host, user=user, password=password, database="test")
cc_query = "select warc_record_id, relevance, text from abortion_corpus_20230125"
cc_data = []
with cc_con:
    with cc_con.cursor() as cursor:
        cursor.execute(cc_query)
        result = cursor.fetchall()
        for r in result:
            cc_data.append({'warc_record_id':r[0],
                            'relevance':r[1],
                            'text':r[2]})
cc_df = pandas.DataFrame(cc_data)

# Now instantiate the object using the dataframe and the TokenFilter
cc_pdo = ParsedDocumentsCommonCrawl(tf, cc_df)
cc_pdo.process_corpus()
"""


class ParsedDocumentsCommonCrawl:
    """
    Class to retrieve and parse corpora from the FourForums data within the Internet Argument Corpus 2.0. This class
    mimics the interfaces provide by the ParsedDocumentsFourForums class.
    """

    def __init__(self, token_filter: TokenFilter, df):
        """
        Initialize the object.

        :param token_filter: a TokenFilter object
        :type token_filter: TokenFilter
        :param df: DataFrame containing a 'text' column which should contain the common crawl texts
        """

        # Add negation extraction to the Spacy NLP pipeline when processing documents
        self.extract_negations = False
        self.tokenize_only = False
        # nlp.add_pipe('sentencizer')
        self.sentencize_only = False
        self.spacy_model = 'en_core_web_lg'

        self.token_filter = token_filter
        self.pre_transform_func = None

        # minimum number of tokens required for us to process a document
        self.length_filter = 0

        # Our dataframe
        self.df = df

        # Number of processes to use with Spacy, defaults to 4
        self.spacy_processes = 4

        # Initialize some Variables
        self.data_structure = {}
        self.raw_text = list()
        self.text = list()
        self.warc_record_id = []
        self.relevance = []
        self.all_docs = []
        self.cas = CorpusAsStems(self.token_filter)
        self.corpusAsStems = self.cas  # a bit more self documenting as an object attribute
        self.target = list()

    def set_pre_transform(self, pre_transform_function):
        """
        Accept a transform function, we will use this function to prefilter the raw text
        before we save it to the data structures.

        :param pre_transform_function: a function that accepts a string, and returns a string
        :return:
        """
        self.pre_transform_func = pre_transform_function

    def process_corpus(self, print_stats=True):
        """
        Process the corpus. This initializes the Spacy model, parses the corpus, gathers stats
        and then returns them. It also prepares the CorpusAsStems index and the all_docs data structure
        which makes the parsed documents available via the all_docs attribute and the target attribute
        which contains the stances of documents.
        """

        # Create the self.all_docs object
        self._process_corpus()  # actually process the corpus .. I guess
        if print_stats:
            self.print_corpus_stats()  # print out some stats on the parse

        # Create a representation of the corpus as stems
        self.cas.extract_stems(self.all_docs)

        # Output basic stats from the stems process
        if print_stats:
            self.cas.print_stats()

    def _process_corpus(self):
        """
        First get the corpus from the datasource, and then parse it using Spacy
        and load the results into the self.all_docs list.

        self.all_docs[] - list of Spacy doc objects, corresponds the order of self.text and self.target
        """
        if self.extract_negations and self.tokenize_only:
            raise RuntimeError("You cannot extract negations if you are only tokenizing the document.")

        # Get the corpus from the data source, this will load the self.text list
        self._get_corpus()

        # Set up Spacy
        torch.set_num_threads(1)  # works around a Spacy bug
        disabled_components = list()
        nlp = spacy.load(self.spacy_model)
        if self.extract_negations:
            ts = termset("en")
            nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})
        elif self.tokenize_only:
            # In Spacy the tokenizer is not a distinct component. So by disabled all components we are telling
            # Spacy to tokenize the text into a document and return it, don't do any further processing.
            disabled_components = [c[0] for c in nlp.pipeline]
        elif self.sentencize_only:
            # disable everything
            disabled_components = [c[0] for c in nlp.pipeline]
            # add the sentencizer back in
            nlp.add_pipe('sentencizer')

        # Parse the Documents and put the parse in self.all_docs
        # print("Loading %s documents" % len(self.text))
        doc_list = nlp.pipe(self.text, n_process=self.spacy_processes,
                            disable=disabled_components)

        for doc in doc_list:
            self.all_docs.append(doc)

    def print_corpus_stats(self):
        """
        Prints some statistics from the corpus we just retrieved.
        """
        print(
            "We grabbed %s documents. Common Crawl documents have no golden standard stance" % len(self.data_structure))
        first_key = list(self.data_structure.keys())[0]
        print("\nHere is our first post from key %s" % first_key)
        print("First 100 Char of Post: ", self.data_structure[first_key][0][0:100])

    def _get_corpus(self):
        """
        Gets the corpus from the data source and does some initial processing to get it into
        some data structures that we can work with later. Creates the following object variables

        self.data_structure{} - dictionary keyed by post_id which holds a tuple (text, stance)
        self.authors_stance1[] - a list with the ids of all authors who take an oppose stance
        self.authors_stance2[] - a list with the ids of all authors who take a support stance
        self.text[] - a list of all the texts in the corpus
        self.target[] - a list of all the targets (stances) for each text, in the same order as self.text
        self.length_filtered_count - the number of documents we filtered out because they were smaller than
                                     self.length_filter
        """

        if len(self.text) > 0:
            raise RuntimeError(
                "We have already processed the corpus during initialization. You cannot initialize again.")

        # Populate the self.data_structure structure, using the warc_record_id as the post id
        for doc_idx, record in enumerate(self.df[['warc_record_id', 'relevance', 'text']].to_dict('records')):
            key = record['warc_record_id']
            text = record['text']
            # Fill in 0 for the stance targeted id and continuous stance
            self.data_structure[key] = (text, 0, 0)
            self.target.append(0)  # for compatibility, all stances are 0

        # Filter out the documents that are too short
        self.length_filtered_count = 0
        for doc_idx, record in enumerate(self.df[['warc_record_id', 'relevance',
                                                  'text']].to_dict('records')):
            # Estimate the length of the text
            length_estimate = len(record['text'].split())

            # Populate our basic data structures needed both to be able to do the Spacy parse
            # and to be able to figure out what the relevance and WARC record id are of each
            # parsed document
            if length_estimate >= self.length_filter:
                if self.pre_transform_func:
                    # If there is a prefilter setup then appy it to the text
                    self.raw_text.append(record['text'])
                    self.text.append(self.pre_transform_func(record['text']))
                else:
                    self.text.append(record['text'])
                    self.raw_text = []

                self.warc_record_id.append(record['warc_record_id'])
                self.relevance.append(record['relevance'])
            else:
                self.length_filtered_count += 1
