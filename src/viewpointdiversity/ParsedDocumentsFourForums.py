import pymysql
import spacy
import torch

from CorpusAsStems import CorpusAsStems


class ParsedDocumentsFourForums:
    """
    Class to retrieve and parse corpora from the FourForums data within the Internet Argument Corpus 2.0
    """

    query = """
        select post.author_id, mturk_author_stance.discussion_id, post.text_id, topic_stance_id_1, 
           a.stance, topic_stance_votes_1, topic_stance_id_2, b.stance, topic_stance_votes_2, text
      from mturk_author_stance join topic_stance a on mturk_author_stance.topic_id = a.topic_id and 
                                                      mturk_author_stance.topic_stance_id_1 = a.topic_stance_id 
                               join topic on topic.topic_id = a.topic_id
                               join topic_stance b on mturk_author_stance.topic_id = b.topic_id and 
                                                      mturk_author_stance.topic_stance_id_2 = b.topic_stance_id
                               join post on post.discussion_id = mturk_author_stance.discussion_id and
                                            post.author_id = mturk_author_stance.author_id 
                               join text on text.text_id = post.text_id 
    where topic = 'gun control'
      -- Unanimous consent by the turkers
      and (topic_stance_votes_1 = 0 or topic_stance_votes_2 = 0)
      -- but there were some votes (there are some rows where both columns are zeros, no votes
      and (topic_stance_votes_1 + topic_stance_votes_2) > 1
      -- limit 200;
    """

    def __init__(self, token_filter, db_name, db_host, db_user, db_password):
        """
        Initialize the object, we actually grab and parse the document on initialization.

        :param token_filter: this is a closure which returns a boolean when passed a Spacy token, it
                             defines whether or not we are considering this token to be valid context.
                             This class does not use it directly, it is just passed to the CorpusAsStems
                             constructor, which uses it to filter out tokens we are not stemming and
                             including in the doc and stem indexes.
        :param db_name: The name of the database to connect to
        :param db_host: The server hosting the database
        :param db_user: The user to use when connecting to the database
        :param db_password: The password to use when connecting to the database
        """
        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.db_user = db_user

        self.token_filter = token_filter

        # minimum number of tokens required for us to process a document
        self.length_filter = 4

        # Initialize some Variables
        self.data_structure = {}
        self.authors_oppose = []
        self.authors_support = []
        self.text = list()
        self.target = list()
        self.all_docs = []

        # Create the self.all_docs object
        self._process_corpus()  # actually process the corpus .. I guess
        self.print_corpus_stats()  # print out some stats on the parse

        # Create a representation of the corpus as stems
        self.cas = CorpusAsStems(self.token_filter)
        self.corpusAsStems = self.cas  # a bit more self documenting as an object attribute
        self.cas.extract_stems(self.all_docs)
        # Output basic stats from the stems process
        self.cas.print_stats()

    def _process_corpus(self):
        """
        First get the corpus from the datasource, and then parse it using Spacy
        and load the results into the self.all_docs list.

        self.all_docs[] - list of Spacy doc objects, corresponds the order of self.text and self.target
        """
        # Get the corpus from the data source, this will load the self.text list
        self._get_corpus()

        # Set up Spacy
        torch.set_num_threads(1)  # works around a Spacy bug
        nlp = spacy.load('en_core_web_lg')

        # Parse the Documents and put the parse in self.all_docs
        print("Loading %s documents" % len(self.text))
        doc_list = nlp.pipe(self.text, n_process=4)
        for doc in doc_list:
            self.all_docs.append(doc)

    def print_corpus_stats(self):
        """
        Prints some statistics from the corpus we just retrieved.
        """
        print("We grabbed %s documents and their stance." % len(self.data_structure))
        print("These posts are from %s distinct authors" % len(set(self.authors_oppose + self.authors_support)))
        first_key = list(self.data_structure.keys())[0]
        print("\nHere is our first post from key %s" % first_key)
        print(self.data_structure[first_key])

        print("{} documents were filtered because they were shorter than {} tokens.".format(self.length_filtered_count,
                                                                                            self.length_filter))

        print(f"Number of posts from supports of strict gun control: {len([x for x in self.target if x == 1])}")
        print("Number of posts from those who oppose strict gun control: {}".format(
            len([x for x in self.target if x == 0])))
        print(f"Number of supporting authors: {len(self.authors_support)}")
        print(f"Number of opposing authors: {len(self.authors_oppose)}")

    def _get_corpus(self):
        """
        Gets the corpus from the data source and does some initial processing to get it into
        some data structures that we can work with later. Creates the following object variables

        self.data_structure{} - dictionary keyed by post_id which holds a tuple (text, stance)
        self.authors_oppose[] - a list with the ids of all authors who take an oppose stance
        self.authors_support[] - a list with the ids of all authors who take a support stance
        self.text[] - a list of all the texts in the corpus
        self.target[] - a list of all the targets (stances) for each text, in the same order as self.text
        self.length_filtered_count - the number of documents we filtered out because they were smaller than
                                     self.length_filter
        """

        if len(self.text) > 0:
            raise RuntimeError(
                "We have already processed the corpus during initialization. You cannot initialize again.")

        # Create our corpus
        connection = pymysql.connect(host=self.db_host, user=self.db_user, password=self.db_password, db=self.db_name,
                                     cursorclass=pymysql.cursors.SSDictCursor)

        # First, Get all the documents and create the indexes
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(self.query)
                result = cursor.fetchone()
                while result is not None:
                    if result['topic_stance_votes_1'] > result['topic_stance_votes_2']:
                        stance_id = result['topic_stance_id_1']
                    else:
                        stance_id = result['topic_stance_id_2']

                    author = result['author_id']
                    # discussion = result['discussion_id']
                    text = result['text']

                    post_id = result['text_id']
                    if stance_id == 3:  # opposes strict gun control
                        stance = 0
                        if author not in self.authors_oppose:
                            self.authors_oppose.append(author)
                    elif stance_id == 2:  # supports strict gun control
                        stance = 1
                        if author not in self.authors_support:
                            self.authors_support.append(author)
                    key = str(post_id)
                    if key in self.data_structure:
                        print("This should never happen, each post_id should be unique")
                        existing_text, existing_stance = self.data_structure[key]
                        if existing_stance != stance:
                            raise RuntimeError("Author changed stance within a discussion")
                        self.data_structure[key] = (existing_text + " " + text, stance)
                    else:
                        self.data_structure[key] = (text, stance)

                    result = cursor.fetchone()

        # Filter out the documents that are too short
        self.length_filtered_count = 0
        rough_lengths = [(x, len(self.data_structure[x][0].split())) for x in self.data_structure]
        for item in rough_lengths:
            if item[1] >= self.length_filter:
                self.text.append(self.data_structure[item[0]][0])
                self.target.append(self.data_structure[item[0]][1])
            else:
                self.length_filtered_count = self.length_filtered_count + 1
