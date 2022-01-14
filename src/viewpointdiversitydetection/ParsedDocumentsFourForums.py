import pymysql
import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
import torch


from viewpointdiversitydetection.TokenFilter import TokenFilter
from viewpointdiversitydetection.CorpusAsStems import CorpusAsStems


class ParsedDocumentsFourForums:
    """
    Class to retrieve and parse corpora from the FourForums data within the Internet Argument Corpus 2.0
    """

    def __init__(self, token_filter: TokenFilter, topic_name, stance_a, stance_b, db_name, db_host, db_user, db_password):
        """
        Initialize the object.

        :param token_filter: a TokenFilter object
        :type token_filter: TokenFilter
        :param topic_name: The name of the topic we are parsing from the FourForums data
        :param stance_a: The first stance to consider
        :param stance_b: The second stance to consider
        :param db_name: The name of the database to connect to
        :param db_host: The server hosting the database
        :param db_user: The user to use when connecting to the database
        :param db_password: The password to use when connecting to the database
        """
        self.topic_name = topic_name
        self.stance_a = stance_a
        self.stance_b = stance_b
        # Our target classes will be strings 'a', and 'b'. The below structure maps those target classes
        # to their descriptions. Note, this is arbitrary. In the SQL below we do not know if the description we are
        # assigning to label 'a' will actually correspond to topic_stance_id_1 or topic_stance_id_2. So it is entirely
        # possible that self.stance_target_id_by_desc[query_result['stance_1']] == 'b'. We account for this in the
        # class code, but if you are debugging or extending something it might throw you for a loop.
        self.stance_target_id_by_desc = {self.stance_a: 'a', self.stance_b: 'b'}
        # This is how much agreement the annotators need to have before
        # we consider a post to have a certain stance. A value of .2 means that
        # 80% of the annotators need to agree for us to use that post and tag it
        # with that stance. Posts that do not reach 80% agreement would be discarded.
        self.stance_agreement_cutoff = .2
        self.num_documents_under_cutoff = 0

        # Add negation extraction to the Spacy NLP pipeline when processing documents
        self.extract_negations = False
        self.tokenize_only = False
        self.spacy_model = 'en_core_web_lg'

        self.query = """
                select post.author_id, mturk_author_stance.discussion_id, post.text_id, topic_stance_id_1, 
               a.stance stance_1, topic_stance_votes_1, topic_stance_id_2, b.stance stance_2,
               topic_stance_votes_2, text.`text`,
               if(topic_stance_votes_1 = 0 || topic_stance_votes_2 = 0, 'unanimous', 'split') consensus,
               (topic_stance_votes_1 / (topic_stance_votes_2 + topic_stance_votes_1)) percent_stance_1,
               (topic_stance_votes_2 / (topic_stance_votes_2 + topic_stance_votes_1)) percent_stance_2
          from mturk_author_stance join topic_stance a on mturk_author_stance.topic_id = a.topic_id and 
                                                  mturk_author_stance.topic_stance_id_1 = a.topic_stance_id 
                                   join topic on topic.topic_id = a.topic_id
                                   join topic_stance b on mturk_author_stance.topic_id = b.topic_id and 
                                                  mturk_author_stance.topic_stance_id_2 = b.topic_stance_id
                                   join post on post.discussion_id = mturk_author_stance.discussion_id and
                                        post.author_id = mturk_author_stance.author_id 
                                   join text on text.text_id = post.text_id 
         where topic.topic = %s
               and a.stance in (%s, %s)
               and b.stance in (%s, %s)
               and topic_stance_votes_1 + topic_stance_votes_2 > 1
        """

        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.db_user = db_user

        self.token_filter = token_filter

        # minimum number of tokens required for us to process a document
        self.length_filter = 4

        # will be set to true of the query results are being limited, false otherwise
        self.is_limited = False

        # Initialize some Variables
        self.data_structure = {}
        self.authors_stance1 = []
        self.authors_stance2 = []
        self.authors_stance_by_desc = {self.stance_a: self.authors_stance1, self.stance_b: self.authors_stance2}
        self.text = list()
        self.target = list()
        self.continuous_target = list()
        self.all_docs = []
        self.cas = CorpusAsStems(self.token_filter)
        self.corpusAsStems = self.cas  # a bit more self documenting as an object attribute

        # Validate the Topic
        self.topics = self._get_valid_topics()
        if topic_name not in self.topics:
            raise ValueError("%s is not a valid topic in the database." % topic_name)

        # Validate the Stance
        self.stances = self._get_valid_stances()
        if stance_a not in self.stances:
            raise ValueError("%s is not a valid stance for topic %s" % (stance_a, topic_name))
        if stance_b not in self.stances:
            raise ValueError("%s is not a valid stance for topic %s" % (stance_b, topic_name))

    def get_stance_label(self, stance_name):
        """
        Return the label being used for the given stance.

        :param stance_name: string containing the stance description
        :return:
        """
        try:
            return self.stance_target_id_by_desc[stance_name]
        except KeyError as e:
            raise ValueError(f"{stance_name} is not a valid stance for this corpus.")

    def set_result_limit(self, result_limit):
        """
        Set a limit on the number of documents we return from the database when parsing texts. Useful for testing.

        :param result_limit: the maximum number of documents to return from the database.
        """
        if type(result_limit) != int:
            raise ValueError("%s is not a valid query limit." % result_limit)
        if self.is_limited:
            raise RuntimeError("The query is already being limited, you cannot set a new limit.")

        self.is_limited = True
        self.query = self.query + "limit %s" % result_limit

    def process_corpus(self):
        """
        Process the corpus. This initializes the Spacy model, parses the corpus, gathers stats
        and then returns them. It also prepares the CorpusAsStems index and the all_docs data structure
        which makes the parsed documents available via the all_docs attribute and the target attribute
        which contains the stances of documents.
        """

        # Create the self.all_docs object
        self._process_corpus()  # actually process the corpus .. I guess
        self.print_corpus_stats()  # print out some stats on the parse

        # Create a representation of the corpus as stems
        self.cas.extract_stems(self.all_docs)

        # Output basic stats from the stems process
        self.cas.print_stats()

    def _get_valid_topics(self):
        """
        Get the valid topics from the database

        :return: a list of valid topics
        """

        # Connect to the database and retrieve all of the valid topics.
        query = """
        select distinct(topic) from topic
        """
        connection = pymysql.connect(host=self.db_host, user=self.db_user, password=self.db_password, db=self.db_name,
                                     cursorclass=pymysql.cursors.SSDictCursor)

        with connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()

        topics = [r['topic'] for r in result]

        return topics

    def _get_valid_stances(self):
        """
        Get the valid stances for the topic in question

        :return: a list of valid stances
        """

        # Connect to the database and retrieve all of the valid topics.
        query = """
        select distinct(st.stance)
          from topic_stance st join topic t on st.topic_id = t.topic_id 
         where t.topic = %s
        """
        connection = pymysql.connect(host=self.db_host, user=self.db_user, password=self.db_password, db=self.db_name,
                                     cursorclass=pymysql.cursors.SSDictCursor)

        with connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (self.topic_name,))
                result = cursor.fetchall()

        stances = [r['stance'] for r in result]

        return stances

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

        # Parse the Documents and put the parse in self.all_docs
        print("Loading %s documents" % len(self.text))
        doc_list = nlp.pipe(self.text, n_process=4, disable=disabled_components)

        for doc in doc_list:
            self.all_docs.append(doc)

    def print_annotated_posts_by_topic(self):
        """
        This method prints out the number of posts in the database that are annotated for stance by topic. It
        gives you an overview of how much useful stance data there is in the FourForums dataset by topic. Note
        that this query is just the upper bounds, some of the stance annotations may not be useful as the annotators
        may be split on the stance of the post.

        :return: dictionary of post counts keyed by topic
        """

        query = """
        select count(distinct(text_id)) as num_posts, topic from ( -- sub select a
            select post.author_id, mturk_author_stance.discussion_id, post.text_id, topic.topic, topic_stance_id_1, 
                   a.stance stance_1, topic_stance_votes_1, topic_stance_id_2, b.stance stance_2, 
                   topic_stance_votes_2, text.`text`,
                   if(topic_stance_votes_1 = 0 || topic_stance_votes_2 = 0, 'unanimous', 'split') consensus,
                   (topic_stance_votes_1 / (topic_stance_votes_2 + topic_stance_votes_1)) percent_stance_1,
                   (topic_stance_votes_2 / (topic_stance_votes_2 + topic_stance_votes_1)) percent_stance_2
            from mturk_author_stance join topic_stance a on mturk_author_stance.topic_id = a.topic_id and 
                                          mturk_author_stance.topic_stance_id_1 = a.topic_stance_id 
                                     join topic on topic.topic_id = a.topic_id
                                     join topic_stance b on mturk_author_stance.topic_id = b.topic_id and 
                                          mturk_author_stance.topic_stance_id_2 = b.topic_stance_id
                                     join post on post.discussion_id = mturk_author_stance.discussion_id and
                                          post.author_id = mturk_author_stance.author_id 
                                     join text on text.text_id = post.text_id 
            where topic_stance_votes_1 + topic_stance_votes_2 > 0) a
        group by topic order by num_posts;
        """
        connection = pymysql.connect(host=self.db_host, user=self.db_user, password=self.db_password, db=self.db_name,
                                     cursorclass=pymysql.cursors.SSDictCursor)

        stats = {}
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchone()
                while results:
                    stats[results['topic']] = results['num_posts']
                    results = cursor.fetchone()

        return stats

    def print_corpus_stats(self):
        """
        Prints some statistics from the corpus we just retrieved.
        """
        print("We grabbed %s documents and their stance." % len(self.data_structure))
        print("These posts are from %s distinct authors" % len(set(self.authors_stance1 + self.authors_stance2)))
        first_key = list(self.data_structure.keys())[0]
        print("\nHere is our first post from key %s" % first_key)
        print("First 100 Char of Post: ", self.data_structure[first_key][0][0:100])
        print("Stance: ", self.data_structure[first_key][1])

        print(f"Inter-annotator agreement cutoff set at {self.stance_agreement_cutoff}")
        msg = "{} documents were rejected because inter-annotator agreement was below the cutoff."
        print(msg.format(self.num_documents_under_cutoff))
        print("{} documents were filtered because they were shorter than {} tokens.".format(self.length_filtered_count,
                                                                                            self.length_filter))

        for desc, target_id in self.stance_target_id_by_desc.items():
            num_posts = len([x for x in self.target if x == target_id])
            print(f"{num_posts} posts from stance {target_id}: {desc}")
        print(f"Number of supporting authors: {len(self.authors_stance2)}")
        print(f"Number of opposing authors: {len(self.authors_stance1)}")

    @staticmethod
    def _calculate_continuous_stance_value(votes_stance_a, votes_stance_b, post_id):
        """
        Private routine that calculates a continuous value for the stance between -1 and 1 based on the
        inter annotator agreement from the corpus using the following formula:

        v_a [=] votes for stance a
        v_b [=] votes for stance b
        cvp [=] continuous viewpoint value
        cvp = (v_b - v_a) / (v_b + v_a)

        a cvp of -1 means 100% inter annotator agreement for viewpoint a, a cvp of 1 means 100% inter annotator
        agreement for viewpoint b.

        If you want to use a different formula create an child class and override this method.

        :param votes_stance_a: number of votes for stance a
        :param votes_stance_b: number of votes for stance b
        :param post_id: the ID of the post in question, used for error messages
        :return: float value between -1 and 1
        """

        # Calculate the continuous stance value. -1 is 100% stance A, 1 is 100% stance B.
        if votes_stance_a + votes_stance_b == 0:
            msg = f"There are no stance votes for {post_id}, can't calculate stance."
            raise RuntimeError(msg)
        v_a = votes_stance_a
        v_b = votes_stance_b
        continuous_stance = (v_b - v_a) / (v_b + v_a)
        return continuous_stance

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

        # Create our corpus
        connection = pymysql.connect(host=self.db_host, user=self.db_user, password=self.db_password, db=self.db_name,
                                     cursorclass=pymysql.cursors.SSDictCursor)

        # First, Get all the documents and create the indexes
        with connection:
            with connection.cursor() as cursor:
                # cursor.execute(self.query, (self.topic_name,))
                cursor.execute(self.query, (self.topic_name, self.stance_a, self.stance_b,
                                            self.stance_a, self.stance_b))
                result = cursor.fetchone()
                while result is not None:
                    stance_vote_difference = result['topic_stance_votes_1'] - result['topic_stance_votes_2']
                    # implement the cutoff. We also don't include the document if it is a tie. That would be fine
                    # for the continuous target value, but wouldn't work at all for the binary stance value.
                    if (result['percent_stance_1'] <= self.stance_agreement_cutoff or
                       result['percent_stance_2'] <= self.stance_agreement_cutoff) and \
                       stance_vote_difference != 0:
                        
                        author = result['author_id']
                        text = result['text']
                        post_id = result['text_id']
    
                        if result['topic_stance_votes_1'] > result['topic_stance_votes_2']:
                            stance_target_id = self.stance_target_id_by_desc[result['stance_1']]
                            authors_stance = self.authors_stance_by_desc[result['stance_1']]
                            if author not in authors_stance:
                                authors_stance.append(author)
                        else:
                            stance_target_id = self.stance_target_id_by_desc[result['stance_2']]
                            authors_stance = self.authors_stance_by_desc[result['stance_2']]
                            if author not in authors_stance:
                                authors_stance.append(author)

                        # Calculate the continuous stance value
                        #
                        # This is confusing ... we don't know, from the SQL code, if result['stance_1'], which is the
                        # string in the database, actually corresponds with
                        # stance_target_id_by_desc[result['stance_1']] == 'a' since the constructor just takes the first
                        # stance string passed in and arbitrarily assigns it to label 'a'. If
                        # stance_target_id_by_desc[result['stance_1']] == 'a' then topic_stance_votes_1 is the votes
                        # for stance a, so we put that argument first and _calculate_continuous_stance_value gives
                        # us < 0.0 for the actual stance a we specified in the constructor.
                        #
                        # However, in the database it could be that result['stance_1'] corresponds to
                        # stance_target_id_by_desc[result['stance_1']] == 'b'. In which case topic_stance_votes_2 are
                        # actually votes for 'a' (i.e. the label in our code is opposite from how the labels are set in
                        # the database, our string labeled stance a corresponds to topic_stance_2.) In that case we need
                        # to switch the order of our arguments to self._calculate_continuous_stance_value so that a
                        # value < 0.0 will be for the string for stance a that was passed to the constructor.
                        if self.stance_target_id_by_desc[result['stance_1']] == 'a':
                            continuous_stance = self._calculate_continuous_stance_value(result['topic_stance_votes_1'],
                                                                                        result['topic_stance_votes_2'],
                                                                                        post_id)
                        else:
                            continuous_stance = self._calculate_continuous_stance_value(result['topic_stance_votes_2'],
                                                                                        result['topic_stance_votes_1'],
                                                                                        post_id)

                        key = str(post_id)
                        if key in self.data_structure:
                            print("In theory this should never happen, each post_id should be unique")
                            existing_text, existing_stance, existing_continuous_stance = self.data_structure[key]
                            if existing_stance != stance_target_id or existing_continuous_stance != continuous_stance:
                                raise RuntimeError("Author changed stance within a discussion")
                            self.data_structure[key] = (existing_text + " " + text, stance_target_id, continuous_stance)
                        else:
                            self.data_structure[key] = (text, stance_target_id, continuous_stance)
                    else:
                        # Keep track of the number of documents we rejected because the annotator confidence
                        # was below the cutoff.
                        self.num_documents_under_cutoff = self.num_documents_under_cutoff + 1

                    result = cursor.fetchone()

        # Filter out the documents that are too short
        self.length_filtered_count = 0
        rough_lengths = [(x, len(self.data_structure[x][0].split())) for x in self.data_structure]
        for item in rough_lengths:
            if item[1] >= self.length_filter:
                self.text.append(self.data_structure[item[0]][0])
                self.target.append(self.data_structure[item[0]][1])
                self.continuous_target.append(self.data_structure[item[0]][2])
            else:
                self.length_filtered_count = self.length_filtered_count + 1
