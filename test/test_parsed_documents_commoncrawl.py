import re
import unittest
import configparser
from src.viewpointdiversitydetection import ParsedDocumentsCommonCrawl
from viewpointdiversitydetection import TokenFilter
import json
import decimal
import pandas


class ParsedDocumentCommonCrawlTest(unittest.TestCase):
    """
    A few tests for the ParsedDocumentsFourForums class. These some of these assume that you have a connection
    to a copy of the Internet Argument Corpus database. Put the database host, username, and password in the
    config.ini file before testing.
    """
    def setUp(self):
        # Load our sample data
        with open('./resources/cc_sample_df.json') as f:
            cc_sample_dict = json.load(f)
            for r in cc_sample_dict:
                # Had to make our decimals a string for the JSON file, convert
                # them back to decimal objects here
                r['relevance'] = decimal.Decimal(r['relevance'])
            cc_sample_df = pandas.DataFrame(cc_sample_dict)

        self.cc_df = cc_sample_df
        self.tf = TokenFilter()

    def test_load_data(self):
        """
        Test to see if the class successfully loads a data frame
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus()

        # Get the number of documents from our test data set
        num_docs = len(self.cc_df)

        # After the parse each of the relevant datastructures
        assert(len(cc_pdo.all_docs) == num_docs)
        assert(len(cc_pdo.warc_record_id) == num_docs)
        assert(len(cc_pdo.relevance) == num_docs)

    def test_pre_transform(self):
        """
        Test the prefilter functionality
        :return:
        """
        def get_transform():
            prolife_pattern_lower = re.compile(r'[p][Rr][Oo][\s-]*[Ll][Ii][Ff][Ee]')
            prochoice_pattern_lower = re.compile(r'[p][Rr][Oo][\s-]*[Cc][Hh][Oo][Ii][Cc][Ee]')
            prolife_pattern_upper = re.compile(r'[P][Rr][Oo][\s-]*[Ll][Ii][Ff][Ee]')
            prochoice_pattern_upper = re.compile(r'[P][Rr][Oo][\s-]*[Cc][Hh][Oo][Ii][Cc][Ee]')

            def transform_text(text):
                """
                t1 = re.sub(prolife_pattern_lower, 'prolife', text)
                t2 = re.sub(prochoice_pattern_lower, 'prochoice', t1)
                t3 = re.sub(prolife_pattern_upper, 'Prolife', t2)
                t4 = re.sub(prochoice_pattern_upper, 'Prochoice', t3)
                """
                t1 = prolife_pattern_lower.sub('prolife', text)
                t2 = prochoice_pattern_lower.sub('prochoice', t1)
                t3 = prolife_pattern_upper.sub('Prolife', t2)
                t4 = prochoice_pattern_upper.sub('Prochoice', t3)
                return t4

            return transform_text

        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.set_pre_transform(get_transform())
        cc_pdo.process_corpus()

        print(cc_pdo.text)
        assert('Pro-life' in cc_pdo.raw_text[0])
        assert('Prolife' in cc_pdo.text[0])

    def test_warc_record_id(self):
        """
        Test the ability to get a documents relevance and worc_record_id from the all_docs index
        :return:
        """
        cc_pdo = ParsedDocumentsCommonCrawl(self.tf, self.cc_df)
        cc_pdo.process_corpus(print_stats=False)

        doc_text = cc_pdo.all_docs[2].text
        doc_relevance = cc_pdo.relevance[2]
        doc_id = cc_pdo.warc_record_id[2]
        df_list = self.cc_df.to_dict('records')

        # Text, relevance, and WARC record ID should line up between the dataframe and the PDO by index
        assert(doc_text == df_list[2]['text'])
        assert(doc_relevance == df_list[2]['relevance'])
        assert(doc_id == df_list[2]['warc_record_id'])




if __name__ == '__main__':
    unittest.main()
