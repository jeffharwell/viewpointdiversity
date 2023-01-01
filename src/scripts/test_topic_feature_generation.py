# General Pieces
import configparser

# Import Viewpoint Diversity Detection Package
import viewpointdiversitydetection as vdd


def get_parsed_corpus(config):
    user = config['InternetArgumentCorpus']['username']
    password = config['InternetArgumentCorpus']['password']
    host = config['InternetArgumentCorpus']['host']
    database = 'fourforums'

    search_terms = ['strict', 'gun', 'control']

    rc = vdd.RawCorpusFourForums('gun control', database, host, user, password)
    rc.stance_a = 'prefers strict gun control'
    rc.stance_b = 'opposes strict gun control'
    corpus_stats = rc.print_stats()

    # Create the corpus. This is doing a lot of work that we don't really need at the moment.
    tf = vdd.TokenFilter()

    pdo = vdd.ParsedDocumentsFourForums(tf, 'gun control', rc.stance_a,
                                        rc.stance_b, database, host, user, password)
    pdo.stance_agreement_cutoff = rc.stance_agreement_cutoff
    label_a = pdo.get_stance_label(rc.stance_a)
    label_b = pdo.get_stance_label(rc.stance_b)
    print(f"{rc.stance_a} => {label_a}, {rc.stance_b} => {label_b}")

    pdo.process_corpus()

    return pdo


def main():
    # Read the configuration (how to get to the database mainly)
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Get the parsed corpus from the database
    pdo = get_parsed_corpus(config)

    # Create the topic vectors
    topic_vector_obj = vdd.TopicFeatureGenerator()
    topic_vector_obj.workers = 7  # good for an 8 core CPU
    topic_vector_obj.debug = True  # we want the debug output
    topic_vector_obj.min_number_topics = 2  # max coherence seems to land at 4 .. which is a bit low
    topic_vector_obj.create_topic_vectors_from_texts(pdo.text)

    print(f"Created a topic vector with {topic_vector_obj.num_topics} topics.")
    print(f"Coherence of the LDA model is {topic_vector_obj.coherence_score}")
    print("First Five Topic Vectors:")
    for tv in topic_vector_obj.topic_vectors[:5]:
        print(f"    {tv}")
    print("Tokens for each topic")
    for t in topic_vector_obj.topics:
        print(f"    {t}")


if __name__ == "__main__":
    main()

