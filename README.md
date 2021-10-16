# VDD Algorithm
Viewpoint Diversity Detection Algorithm

# Dependencies

## Packages

* nltk
* spacy
* sklearn
* gensim
* numpy
* matplotlib
* PyMySQL (For connecting to a copy of the Internet Argument Corpus database)
* pandas
* textblob (for sentiment analysis)
* torch (required by Spacy)

## Gensim

Gensim is going to download some very big files. If you want to control
where those files go set the following environment commands in your Python
script. For example if you have a directory '/bigtmp' where you want all 
Gensim data to go set you environmental variables as follows:

For running from a BASH shell:

    export GENSIM_DATA_DIR=/bigtmp
    export TEMP=/bigtmp

Here is how you would set it if calling the module from a Jupyter notebook:

    %set_env GENSIM_DATA_DIR=/bigtmp
    %set_env TEMP=/bigtmp

## NLTK Stopwords
You will need the nltk package install and the stopwords downloaded

    # python
    >>> import nltk
    >>> nltk.download('stopwords')
    >>> nltk.download('vader_lexicon')

## Spacy
You will need the package spacy installed and run the following to download the parsing model:

    $ python -m spacy download en_core_web_lg

# Other
The packaging tutorial I followed to set this whole thing up.
https://packaging.python.org/tutorials/packaging-projects/

# To Build
python -m build

# To Install Locally
python -m pip install ./dist/viewpointdiversitydetection-0.0.1-py3-none-any.whl

# To Uninstall
python -m pip uninstall viewpointdiversitydetection

# Usage

The below is a basic usage of the package which will create a set of feature vectors
for the climate control debate texts within the Internet Argument Corpus 2.0 using a small
pre-trained word2vec model. 

    import viewpointdiversitydetection as vdd
    import configparser
    import gensim.downloader as api
    from nltk.corpus import stopwords
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    stop_words = set(stopwords.words('english'))
    stop_words = [s for s in stop_words if s not in ['no', 'nor', 'not']]  # I want negations
    
    def token_filter(spacy_token):
        if not spacy_token.is_space and not spacy_token.is_punct and spacy_token.text.lower() not in stop_words:
            return True
        else:
            return False
    
    user = config['InternetArgumentCorpus']['username']
    password = config['InternetArgumentCorpus']['password']
    host = config['InternetArgumentCorpus']['host']
    database = 'fourforums'
    
    pdo = vdd.ParsedDocumentsFourForums(token_filter, 'climate change', 'humans not responsible',
                                    'humans responsible', database, host, user, password)
    pdo.set_result_limit(500)
    pdo.process_corpus()
    search_terms = ['human', 'responsibleb', 'climate', 'change']
    
    fk = vdd.FindCharacteristicKeywords(pdo)
    print("\n-- Extracted nouns related to the search terms")
    
    #
    # Now Create the Feature Vector Object
    #
    vector_model = api.load('glove-wiki-gigaword-50')
    context_size = 6
    
    fvt = vdd.FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
    fvt.create_feature_vectors_and_targets()
    
    print(f"Created {len(fvt.feature_vectors)} feature vectors.")
