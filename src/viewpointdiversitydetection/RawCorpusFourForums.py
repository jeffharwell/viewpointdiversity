import pymysql


class RawCorpusFourForums:

    def __init__(self, topic_name, db_name, db_host, db_user, db_password):
        """
        Object which allows listing of the valid stances for a topic and also pulls
        basic statistics on a raw corpus.

        :param topic_name: The name of the topic we are parsing from the FourForums data
        :param db_name: The name of the database to connect to
        :param db_host: The server hosting the database
        :param db_user: The user to use when connecting to the database
        :param db_password: The password to use when connecting to the database
        """

        self.topic_name = topic_name
        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.db_user = db_user

        self.stance_a = None
        self.stance_b = None
        # This is how much agreement the annotators need to have before
        # we consider a post to have a certain stance. A value of .2 means that
        # 80% of the annotators need to agree for us to use that post and tag it
        # with that stance. Posts that do not reach 80% agreement would be discarded.
        self.stance_agreement_cutoff = .2

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

    def get_valid_stances(self):
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

    def print_stats(self):
        """
        Print some stats about the corpus that we extract straight from the database.

        :return: dictionary with some basic stats
        """

        if not self.stance_a or not self.stance_b:
            raise RuntimeError("You must define stance_a and stance_b before generating corpus statistics.")

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
                cursor.execute(query_stance_coded_posts, (self.topic_name, self.stance_a, self.stance_b,
                                                          self.stance_a, self.stance_b))
                results = cursor.fetchall()
                stats['posts_with_stances'] = results[0]['c']
                # Corpus Stats by Stance
                unanimous_posts = {self.stance_a: 0, self.stance_b: 0}
                split_posts = {self.stance_a: 0, self.stance_b: 0}
                all_posts = {self.stance_a: 0, self.stance_b: 0}
                authors = {self.stance_a: [], self.stance_b: []}
                post_length = {self.stance_a: [], self.stance_b: []}
                cursor.execute(self.query, (self.topic_name, self.stance_a, self.stance_b,
                                            self.stance_a, self.stance_b))
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
        stats['stance_a_description'] = self.stance_a
        stats['stance_b_description'] = self.stance_b
        stats['stance_agreement_cutoff'] = self.stance_agreement_cutoff
        stats['authors_with_usable_stances'] = len(authors[self.stance_a]) + len(authors[self.stance_b])
        stats['authors_stance_a'] = len(authors[self.stance_a])
        stats['authors_stance_b'] = len(authors[self.stance_b])

        # Did we have any authors who switched stances on us between discussions
        # & is the set intersection operator
        authors_who_switched_stances = set(authors[self.stance_a]) & set(authors[self.stance_b])
        stats['authors_who_switched_stances'] = len(authors_who_switched_stances)

        if len(discussions) == 0:
            print(posts_considered)
            raise RuntimeError("No discussions with stance were found for topic %s" % self.topic_name)

        if len(post_length[self.stance_a]) == 0 or len(post_length[self.stance_b]) == 0:
            msg = f"No posts with a usable stance found for either stance '{self.stance_a}' or '{self.stance_b}'"
            raise RuntimeError(msg)

        # Number of discussions
        stats['number_of_discussions_with_stance'] = len(discussions)
        stats['avg_length_of_discussions_with_stance'] = sum([d for d in discussion_length.values()]) / len(discussions)

        # How many posts have a consensus stance
        stats['posts_unanimous_stance_a'] = unanimous_posts[self.stance_a]
        stats['posts_unanimous_stance_b'] = unanimous_posts[self.stance_b]
        stats['posts_unanimous_total'] = unanimous_posts[self.stance_a] + unanimous_posts[self.stance_b]

        # How many posts have a split stance
        stats['posts_split_stance_a'] = split_posts[self.stance_a]
        stats['posts_split_stance_b'] = split_posts[self.stance_b]
        stats['posts_split_total'] = split_posts[self.stance_a] + split_posts[self.stance_b]
        stats['inter-annotator_cutoff'] = self.stance_agreement_cutoff
        stats['posts_with_usable_stance'] = stats['posts_unanimous_total'] + stats['posts_split_total']

        total_post_len_s1 = sum(post_length[self.stance_a])
        total_post_len_s2 = sum(post_length[self.stance_b])
        total_post_len = total_post_len_s1 + total_post_len_s2
        num_post_s1 = len(post_length[self.stance_a])
        num_post_s2 = len(post_length[self.stance_b])
        total_num_post = num_post_s1 + num_post_s2
        stats['avg_length_of_post_stance_a'] = round(total_post_len_s1 / num_post_s1, 4)
        stats['avg_length_of_post_stance_b'] = round(total_post_len_s2 / num_post_s2, 4)
        stats['avg_length_of_post_with_stance'] = round(total_post_len / total_num_post, 4)
        stats['min_length_of_post_with_stance'] = min(post_length[self.stance_a] + post_length[self.stance_b])
        stats['max_length_of_post_with_stance'] = max(post_length[self.stance_a] + post_length[self.stance_b])

        longest_key_string = max([len(k) for k in stats.keys()])
        longest_value_string = max([len(str(v)) for v in stats.values()])

        title = self.topic_name + " Topic Stats"
        print(f"| {title.ljust(longest_key_string)} ||")
        print("| "+"-"*longest_key_string + " | " + "-"*longest_value_string + " |")
        for label, value in stats.items():
            print(f"| {label.ljust(longest_key_string)} | {str(value).ljust(longest_value_string)} |")

        return stats
