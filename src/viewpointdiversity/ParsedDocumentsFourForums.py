import pymysql
import spacy
import torch

from CorpusAsStems import CorpusAsStems


class ParsedDocumentsFourForums:
    """
    Class to retrieve and parse corpora from the FourForums data within the Internet Argument Corpus 2.0
    """

    def __init__(self, token_filter, topic_name, stance1, stance2, db_name, db_host, db_user, db_password):
        """
        Initialize the object, we actually grab and parse the document on initialization.

        :param token_filter: this is a closure which returns a boolean when passed a Spacy token, it
                             defines whether or not we are considering this token to be valid context.
                             This class does not use it directly, it is just passed to the CorpusAsStems
                             constructor, which uses it to filter out tokens we are not stemming and
                             including in the doc and stem indexes.
        :param topic_name: The name of the topic we are parsing from the FourForums data
        :param stance1: The first stance to consider
        :param stance2: The second stance to consider
        :param db_name: The name of the database to connect to
        :param db_host: The server hosting the database
        :param db_user: The user to use when connecting to the database
        :param db_password: The password to use when connecting to the database
        """
        self.topic_name = topic_name
        self.stance1 = stance1
        self.stance2 = stance2
        # Our target classes will be integers 1, and 2. This structure maps those target classes
        # to their descriptions
        self.stance_target_id_by_desc = {self.stance1: 1, self.stance2: 2}
        # This is how much agreement the annotators need to have before
        # we consider a post to have a certain stance. A value of .2 means that
        # 80% of the annotators need to agree for us to use that post and tag it
        # with that stance. Posts that do not reach 80% agreement would be discarded.
        self.stance_agreement_cutoff = .2
        self.num_documents_under_cutoff = 0

        self.query = """
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
        where topic = %s
          -- Unanimous consent by the turkers
          and (topic_stance_votes_1 = 0 or topic_stance_votes_2 = 0)
          -- but there were some votes (there are some rows where both columns are zeros, no votes
          and (topic_stance_votes_1 + topic_stance_votes_2) > 1
          -- limit 200;
        """
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
        self.authors_stance_by_desc = {self.stance1: self.authors_stance1, self.stance2: self.authors_stance2}
        self.text = list()
        self.target = list()
        self.all_docs = []
        self.cas = CorpusAsStems(self.token_filter)
        self.corpusAsStems = self.cas  # a bit more self documenting as an object attribute

        # Validate the Topic
        self.topics = self._get_valid_topics()
        if topic_name not in self.topics:
            raise ValueError("%s is not a valid topic in the database." % topic_name)

        # Validate the Stance
        self.stances = self._get_valid_stances()
        if stance1 not in self.stances:
            raise ValueError("%s is not a valid stance for topic %s" % (stance1, topic_name))
        if stance2 not in self.stances:
            raise ValueError("%s is not a valid stance for topic %s" % (stance2, topic_name))

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

    def print_raw_corpus_stats(self):
        """
        Print some stats about the corpus that we extract straight from the database.

        :return: dictionary with some basic stats
        """
        query_num_discussions = """
            select count(*) c
              from discussion d join discussion_topic dt on dt.discussion_id = d.discussion_id 
                                join topic t on t.topic_id = dt.topic_id
             where t.topic = %s
        """
        query_num_posts = """
            select count(*) c
                from discussion d join discussion_topic dt on dt.discussion_id = d.discussion_id 
                    join topic t on t.topic_id = dt.topic_id
                    join post p on p.discussion_id = dt.discussion_id 
                where t.topic = %s
        """
        query_num_authors = """
            select count(distinct(p.author_id)) c
                from discussion d join discussion_topic dt on dt.discussion_id = d.discussion_id 
                    join topic t on t.topic_id = dt.topic_id
                    join post p on p.discussion_id = d.discussion_id
                where t.topic = %s
        """
        query_stance_coded_posts = """
        select count(*) c 
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
          and topic_stance_votes_1 + topic_stance_votes_2 > 0;
        """
        query_posts_stances_authors = """
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

        # Create our corpus
        connection = pymysql.connect(host=self.db_host, user=self.db_user, password=self.db_password, db=self.db_name,
                                     cursorclass=pymysql.cursors.SSDictCursor)
        stats = {}

        # First, Get all the documents and create the indexes
        with connection:
            with connection.cursor() as cursor:
                # Number of discussions in the database for this topic
                cursor.execute(query_num_discussions, (self.topic_name,))
                results = cursor.fetchall()
                stats['discussion_topics_in_db'] = results[0]['c']
                # Number of posts in the database for this topic
                cursor.execute(query_num_posts, (self.topic_name,))
                results = cursor.fetchall()
                stats['posts_in_db'] = results[0]['c']
                # Number of authors
                cursor.execute(query_num_authors, (self.topic_name,))
                results = cursor.fetchall()
                stats['authors_in_db'] = results[0]['c']
                # Number of posts coded for the stances we are looking at
                cursor.execute(query_stance_coded_posts, (self.topic_name, self.stance1, self.stance2,
                                                          self.stance1, self.stance2))
                results = cursor.fetchall()
                stats['posts_with_stances'] = results[0]['c']
                # Corpus Stats by Stance
                unanimous_posts = {self.stance1: 0, self.stance2: 0}
                split_posts = {self.stance1: 0, self.stance2: 0}
                all_posts = {self.stance1: 0, self.stance2: 0}
                authors = {self.stance1: [], self.stance2: []}
                post_length = {self.stance1: [], self.stance2: []}
                #cursor.execute(query_posts_stances_authors, (self.topic_name, self.stance1, self.stance2,
                #                                             self.stance1, self.stance2))
                cursor.execute(self.query, (self.topic_name, self.stance1, self.stance2,
                                            self.stance1, self.stance2))
                discussions = []
                discussion_length = {}
                posts_considered = 0

                result = cursor.fetchone()
                while result:
                    posts_considered = posts_considered + 1
                    if result['percent_stance_1'] <= self.stance_agreement_cutoff or \
                       result['percent_stance_2'] <= self.stance_agreement_cutoff:
                        # Determine the stance of this row
                        if result['percent_stance_1'] > result['percent_stance_2']:
                            result_stance = result['stance_1']
                        else:
                            result_stance = result['stance_2']

                        # Populate the unanimous and split post counts
                        if result['consensus'] == 'unanimous':
                            unanimous_posts[result_stance] = unanimous_posts[result_stance] + 1
                        if result['consensus'] == 'split':
                            split_posts[result_stance] = split_posts[result_stance] + 1

                        # Keep track of the total number of posts as well
                        all_posts[result_stance] = all_posts[result_stance] + 1

                        # Keep track of the number of authors who hold this stance
                        if result['author_id'] not in authors[result_stance]:
                            authors[result_stance].append(result['author_id'])

                        # Discussion and Discussion Length
                        if not result['discussion_id'] in discussions:
                            discussions.append(result['discussion_id'])
                            discussion_length[result['discussion_id']] = 1
                        else:
                            discussion_length[result['discussion_id']] = discussion_length[result['discussion_id']] + 1

                        # Post Length
                        length_of_post = len(result['text'].split(" "))
                        post_length[result_stance].append(length_of_post)

                    result = cursor.fetchone()

        #
        # Now compute the various stats and add then to our stats object
        #
        stats['stance_1_description'] = self.stance1
        stats['stance_2_description'] = self.stance2
        stats['stance_agreement_cutoff'] = self.stance_agreement_cutoff
        stats['authors_with_usable_stances'] = len(authors[self.stance1]) + len(authors[self.stance2])
        stats['authors_stance_one'] = len(authors[self.stance1])
        stats['authors_stance_two'] = len(authors[self.stance2])

        # Did we have any authors who switched stances on us between discussions
        # & is the set intersection operator
        authors_who_switched_stances = set(authors[self.stance1]) & set(authors[self.stance2])
        stats['authors_who_switched_stances'] = len(authors_who_switched_stances)

        if len(discussions) == 0:
            print(posts_considered)
            raise RuntimeError("No discussions with stance were found for topic %s" % self.topic_name)

        if len(post_length[self.stance1]) == 0 or len(post_length[self.stance2]) == 0:
            msg = f"No posts with a usable stance found for either stance '{self.stance1}' or '{self.stance2}'"
            raise RuntimeError(msg)

        # Number of discussions
        stats['number_of_discussions_with_stance'] = len(discussions)
        stats['avg_length_of_discussions_with_stance'] = sum([d for d in discussion_length.values()]) / len(discussions)

        # How many posts have a consensus stance
        stats['posts_unanimous_stance_one'] = unanimous_posts[self.stance1]
        stats['posts_unanimous_stance_two'] = unanimous_posts[self.stance2]
        stats['posts_unanimous_total'] = unanimous_posts[self.stance1] + unanimous_posts[self.stance2]

        # How many posts have a split stance
        stats['posts_split_stance_one'] = split_posts[self.stance1]
        stats['posts_split_stance_two'] = split_posts[self.stance2]
        stats['posts_split_total'] = split_posts[self.stance1] + split_posts[self.stance2]

        stats['posts_with_usable_stance'] = stats['posts_unanimous_total'] + stats['posts_split_total']

        total_post_len_s1 = sum(post_length[self.stance1])
        total_post_len_s2 = sum(post_length[self.stance2])
        total_post_len = total_post_len_s1 + total_post_len_s2
        num_post_s1 = len(post_length[self.stance1])
        num_post_s2 = len(post_length[self.stance2])
        total_num_post = num_post_s1 + num_post_s2
        stats['avg_length_of_post_stance_one'] = round(total_post_len_s1 / num_post_s1, 4)
        stats['avg_length_of_post_stance_two'] = round(total_post_len_s2 / num_post_s2, 4)
        stats['avg_length_of_post_with_stance'] = round(total_post_len / total_num_post, 4)
        stats['min_length_of_post_with_stance'] = min(post_length[self.stance1] + post_length[self.stance2])
        stats['max_length_of_post_with_stance'] = max(post_length[self.stance1] + post_length[self.stance2])

        longest_key_string = max([len(k) for k in stats.keys()])
        longest_value_string = max([len(str(v)) for v in stats.values()])

        title = self.topic_name + " Topic Stats"
        print(f"| {title.ljust(longest_key_string)} ||")
        print("| "+"-"*longest_key_string + " | " + "-"*longest_value_string + " |")
        for label, value in stats.items():
            print(f"| {label.ljust(longest_key_string)} | {str(value).ljust(longest_value_string)} |")

        return stats

    def print_corpus_stats(self):
        """
        Prints some statistics from the corpus we just retrieved.
        """
        print("We grabbed %s documents and their stance." % len(self.data_structure))
        print("These posts are from %s distinct authors" % len(set(self.authors_stance1 + self.authors_stance2)))
        first_key = list(self.data_structure.keys())[0]
        print("\nHere is our first post from key %s" % first_key)
        print(self.data_structure[first_key])

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
                cursor.execute(self.query, (self.topic_name, self.stance1, self.stance2,
                                            self.stance1, self.stance2))
                result = cursor.fetchone()
                while result is not None:
                    # implement the cutoff
                    if result['percent_stance_1'] <= self.stance_agreement_cutoff or \
                       result['percent_stance_2'] <= self.stance_agreement_cutoff:
                        
                        author = result['author_id']
                        text = result['text']
    
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
    
                        post_id = result['text_id']
                        key = str(post_id)
                        if key in self.data_structure:
                            print("In theory this should never happen, each post_id should be unique")
                            existing_text, existing_stance = self.data_structure[key]
                            if existing_stance != stance_target_id:
                                raise RuntimeError("Author changed stance within a discussion")
                            self.data_structure[key] = (existing_text + " " + text, stance_target_id)
                        else:
                            self.data_structure[key] = (text, stance_target_id)
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
            else:
                self.length_filtered_count = self.length_filtered_count + 1
