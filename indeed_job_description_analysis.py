from __future__ import division
import requests
from lxml import html
import time
from bs4 import BeautifulSoup
import re
from time import sleep
from collections import Counter, defaultdict
from nltk.corpus import stopwords
import pandas as pd
from collections import Counter
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def text_cleaner(website):
    # copied and slightly modified from https://jessesw.com/Data-Science-Skills/
    '''
    This function just cleans up the raw html so that I can look at it.
    Inputs: a URL to investigate
    Outputs: Cleaned text only
    '''
    try:
        site = requests.get(website, timeout = 10).text.encode('utf-8') # Connect to the job posting
    except (KeyboardInterrupt):
        raise
    except:
        return []   # Need this in case the website isn't there anymore or some other weird connection problem
    soup_obj = BeautifulSoup(site, "lxml") # Get the html from the site
    for script in soup_obj(["script", "style"]):
        script.extract() # Remove these two elements from the BS4 object
    text = soup_obj.get_text() # Get the text from this
    lines = (line.strip() for line in text.splitlines()) # break into lines
    chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) # break multi-headlines into a line each

    def chunk_space(chunk):
        chunk_out = chunk + ' ' # Need to fix spacing issue
        return chunk_out

    text = ''.join(chunk_space(chunk) for chunk in chunks if chunk).encode('utf-8') # Get rid of all blank lines and ends of line
    # Now clean out all of the unicode junk (this line works great!!!)
    try:
        text = text.decode('unicode_escape').encode('ascii', 'ignore') # Need this as some websites aren't formatted
    except (KeyboardInterrupt):
        raise
    except:                                                            # in a way that this works, can occasionally throw
        return []                                                     # an exception
    text = re.sub("[^a-zA-Z0-9+#-]"," ", text)  # Now get rid of any terms that aren't words (include 3 for d3.js)
    text = text.lower().split()  # Go to lower case and split them apart
    stop_words = set(stopwords.words("english")) # Filter out any stop words
    text = [w for w in text if not w in stop_words]
    return text

def get_unique_links(search_terms, title_only = False, location = "", max_results = 0):
    """
    Input: Terms to search for on indeed. Supports quotations.
    Output: Dataframe of all job titles and links returned for those search terms
    """
    search_term = '+'.join(search_terms.split(' '))
    loc = '+'.join(location.split(' '))

    if (title_only):
        base_url = 'http://www.indeed.com/jobs?as_ttl={}&l={}&limit=100&start='.format(search_term, loc)
    else:
        base_url = 'http://www.indeed.com/jobs?q={}&l={}&limit=100&start='.format(search_term, loc)

    # get max number of results
    initial = html.fromstring(requests.get(base_url+'0', timeout=10).text)
    search_count = int(initial.xpath('//div[@id="searchCount"]/text()')[0].split()[-1].replace(',',''))

    all_links = []
    all_titles = []
    for i in range(0,search_count,100):
        page = requests.get(base_url + str(i), timeout=10)
        tree = html.fromstring(page.text)
        titles = tree.xpath('//a[@data-tn-element="jobTitle"]/@title')
        links = tree.xpath('//a[@data-tn-element="jobTitle"]/@href')
        for link in links:
            all_links.append(link)
        for title in titles:
            all_titles.append(title)
        print 'Reading Indeed page {} of {}'.format(i/100,search_count/100)
        if (search_count >= 101 and ('Next' not in tree.xpath('//span[@class="np"]/text()')[-1])):
            break
        if (max_results and max_results <= i+100):
            break
        time.sleep(1)
    # weed out duplicate links
    df = pd.DataFrame({'title':all_titles,'link':all_links})
    df = df[~df['link'].str.contains('pagead')]
    df.drop_duplicates(inplace=True)
    return df


if __name__=='__main__':
    search_terms  = ['"data scientist"', '"data analyst"','"data engineer"', '"machine learning"', '"business intelligence"', '"business analytics"']
    one_gram_tf = [[] for i in search_terms]
    two_gram_tf = [[] for i in search_terms]
    #three_grams = [[] for i in search_terms]

    for i,term in enumerate(search_terms):
        one_counter = Counter()
        two_counter = Counter()
        #three_counter = Counter()
        print ("Getting results for search term '{}'".format(term))
        unique_links = get_unique_links(term, title_only = True, max_results = 500)['link'].values
        for j,link in enumerate(unique_links):
            pairs = set()
            #triplets = set()
            doc = text_cleaner('http://www.indeed.com'+link)
            doc_count = 0
            if doc:
                doc_count += 1
                print 'Getting job number {}'.format(j)
                one_counter.update(set(doc))
                for k in range(len(doc)-1):
                    pairs.add((doc[k],doc[k+1]))
                two_counter.update(pairs)
            else:
                print 'Failed to retrieve job number {}'.format(j)
        # term frequency
        one_gram_tf[i] = {key:val/doc_count for key,val in one_counter.items()}
        two_gram_tf[i] = {key:val/doc_count for key,val in two_counter.items()}

        # save files inside for loop in order to retain finished search terms in case of failure
        with open('one_gram_tf.pkl', 'wb') as myfile:
            pickle.dump(one_gram_tf, myfile)
        with open('two_gram_tf.pkl', 'wb') as myfile:
            pickle.dump(two_gram_tf, myfile)

    # count how many searches the term occurs in for averaging
    one_gram_counter = Counter()
    two_gram_counter = Counter()
    for job_title in one_gram_tf:
        one_gram_counter.update(job_title.keys())
    for job_title in two_gram_tf:
        two_gram_counter.update(job_title.keys())

    # get average counts for comparison
    # probably a better way to do this with map reduce.
    one_gram_avg = defaultdict(int)
    two_gram_avg = defaultdict(int)
    for job_title in one_gram_tf:
        for key,val in job_title.iteritems():
            one_gram_avg[key] += val/one_gram_counter[key]
    for job_title in two_gram_tf:
        for key,val in job_title.iteritems():
            two_gram_avg[key] += val/two_gram_counter[key]

    # term frequency - relative term frequency
    # need to experiment with other comparison formulas to see which ones work best [(tf - rtf), tf*(tf - rtf), (tf / rtf), tf*(tf / rtf), weighted combinations?]
    tf_rtf_one = [dict() for i in one_gram_tf]
    for i in range(len(one_gram_tf)):
        for key in one_gram_tf[i].keys():
            tf_rtf_one[i][key] = (one_gram_tf[i][key] * one_gram_tf[i][key] / one_gram_avg[key])

    tf_rtf_two = [dict() for i in two_gram_tf]
    for i in range(len(two_gram_tf)):
        for key in two_gram_tf[i].keys():
            tf_rtf_two[i][key] = (two_gram_tf[i][key] * two_gram_tf[i][key] / two_gram_avg[key])
