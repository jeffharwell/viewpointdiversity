#!/usr/bin/python

import os

# Our SVM and some associated items
import pickle

# General Pieces
import configparser

# Import Viewpoint Diversity Detection Package
import src.viewpointdiversitydetection as vdd

# Word2Vec library
import gensim.downloader as api

"""
Trying something different. Load a single vector model and then create the features for multiple corpora
and save those in a database table.
"""


class CorpusDefinition:
    """
    All the information that we need to create process a
    corpus and save the features in a table.
    """

    def __init__(self, topic):
        """
        Initialize the object
        :param topic: string with the name of the topic
        """
        self.topic = topic
        self.stance_a = None
        self.stance_b = None
        self.search_terms = None

    def set_stance(self, stance_a, stance_b):
        """
        Set the stances.

        :param stance_a: string, text of stance a
        :param stance_b: string, text of stance b
        """

        self.stance_a = stance_a
        self.stance_b = stance_b

    def set_search_terms(self, search_terms):
        """
        Set the search terms to use when extracting the contexts.

        :param search_terms: the search terms to use
        """
        self.search_terms = search_terms


def main():
    gun_control_definition = CorpusDefinition('gun control')
    evolution_definition = CorpusDefinition('evolution')
    abortion_definition = CorpusDefinition('abortion')
    existence_of_god_definition = CorpusDefinition('existence of God')

    gun_control_definition.set_stance('prefers strict gun control',
                                      'opposes strict gun control')
    evolution_definition.set_stance('evolution occurs via purely natural mechanisms',
                                    'evolution involves more than purely natural mechanisms (intelligent design)')
    abortion_definition.set_stance('pro-life', 'pro-choice')
    existence_of_god_definition.set_stance('atheist', 'theist')

    gun_control_definition.set_search_terms(['strict', 'gun', 'control'])
    evolution_definition.set_search_terms(['evolution', 'natural', 'mechanism', 'intelligent', 'design'])
    abortion_definition.set_search_terms(['abortion', 'pro-life', 'pro-choice'])
    existence_of_god_definition.set_search_terms(['atheist', 'theist', 'God', 'exist'])

    corpus_definitions = [gun_control_definition, evolution_definition,
                          abortion_definition, existence_of_god_definition]

    vector_models = [{'short_name': 'fasttext-wiki', 'api_load': 'fasttext-wiki-news-subwords-300'},
                     {'short_name': 'word2vec-google-news', 'api_load': 'word2vec-google-news-300'},
                     {'short_name': 'glove-gigaword', 'api_load': 'glove-wiki-gigaword-300'}]

    vm_api_name = vector_models[2]['api_load']
    vm_short_name = vector_models[2]['short_name']
    print(f"Loading vector model {vm_short_name}")
    vector_model = api.load(vm_api_name)

    for cd in corpus_definitions:
        #
        # Load the corpus from the database
        #
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'
        rc = vdd.RawCorpusFourForums(cd.topic, database, host, user, password)
        rc.stance_a = cd.stance_a
        rc.stance_b = cd.stance_b
        rc.print_stats()

        #
        # Define the tokens we are interested in and then retrieve and pre-process our corpus
        #
        tf = vdd.TokenFilter()

        #
        # Parse the corpus
        #

        # Store and Load this Structure
        # filename: {topic}_pdo.pickle
        pdo_pickle_name = f"{cd.topic}_pdo.pickle"
        if os.path.exists(pdo_pickle_name):
            print("Loading Parsed Documents from File")
            infile = open(pdo_pickle_name, 'rb')
            pdo = pickle.load(infile)
            infile.close()
        else:
            print("Parsing Documents")
            pdo = vdd.ParsedDocumentsFourForums(tf, cd.topic, rc.stance_a,
                                                rc.stance_b, database, host, user, password)
            pdo.stance_agreement_cutoff = rc.stance_agreement_cutoff
            label_a = pdo.get_stance_label(rc.stance_a)
            label_b = pdo.get_stance_label(rc.stance_b)
            print(f"{rc.stance_a} => {label_a}, {rc.stance_b} => {label_b}")

            pdo.process_corpus()

            outfile = open(pdo_pickle_name, 'wb')
            pickle.dump(pdo, outfile)
            outfile.close()

        #
        # Get the Contexts
        #

        # Store and Load this Structure as well
        # should just be able to store the related terms here
        # filename: {topic}_related_terms.pickle
        pickle_name = f"{cd.topic}_related_terms.pickle"
        if os.path.exists(pickle_name):
            print("Loading Related Terms from File")
            infile = open(pickle_name, 'rb')
            related_terms = pickle.load(infile)
            infile.close()
        else:
            print("Finding Related Terms")
            fk = vdd.FindCharacteristicKeywords(pdo)
            related_terms = fk.get_unique_nouns_from_term_context(cd.search_terms)
            print(f"{len(cd.search_terms)} search terms and {len(related_terms)} related terms found for corpus.")

            outfile = open(pickle_name, 'wb')
            pickle.dump(related_terms, outfile)
            outfile.close()
            del fk

        #
        # Create the Feature Vector from the Contexts
        #
        pickle_name = f"{cd.topic}_{vm_short_name}_features.pickle"
        if os.path.exists(pickle_name):
            print("Loading Features from File")
            infile = open(pickle_name, 'rb')
            fvt = pickle.load(infile)
            infile.close()
        else:
            print("Creating Features")
            context_size = 6
            fvt = vdd.FeatureVectorsAndTargets(pdo, vector_model, cd.search_terms, related_terms, context_size)
            fvt.create_feature_vectors_and_targets()
            print(f"Created {len(fvt.feature_vectors)} feature vectors.")

            outfile = open(pickle_name, 'wb')
            pickle.dump(fvt, outfile)
            outfile.close()

        print(f"Feature vector size is {len(fvt.feature_vectors[0])}.")


if __name__ == '__main__':
    main()
