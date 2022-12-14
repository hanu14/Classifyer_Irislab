import operator
import re
import os
import json
import logging
from collections import Counter
import urllib

from tqdm import tqdm
import colorlog
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from unidecode import unidecode

#####################
# Hyperparameters
#####################
CONTEXT_LENGTH = 100
HASHTAG_VOCAB_SIZE = 40000
DATA_ROOT_PATH = '../data_yfcc'

# For dataset
HASHTAG_TRAIN_JSON_FNAME = os.path.join(
    DATA_ROOT_PATH, 'json', 'hashtag_train.json'
)
HASHTAG_TEST_JSON_FNAME = os.path.join(
    DATA_ROOT_PATH, 'json', 'hashtag_test.json'
)


HASHTAG_OUTPUT_PATH = os.path.join(DATA_ROOT_PATH, 'hashtag_dataset')

HASHTAG_VOCAB_FNAME = os.path.join(
    HASHTAG_OUTPUT_PATH, '%d.vocab' % (HASHTAG_VOCAB_SIZE)
)

# For vocaulary
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# For tokenization
try:
  # UCS-4
  EMOTICON = re.compile(u'(([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF]))')
except Exception as e:
  # UCS-2
  EMOTICON = re.compile(u'(([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF]))')
NOT_EMOTICON = re.compile(r'(\\U([0-9A-Fa-f]){8})|(\\u([0-9A-Fa-f]){4})')

# Regular expressions used to tokenize
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d+")
_URL_RE = re.compile(r'(?i)\b((?:http[s]?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>"\']+))')
_HREF_RE = re.compile('<a href="(.*?)".*>(.*)</a>')


def sort_dict(dic):
  # Sort by alphabet
  sorted_pair_list = sorted(dic.items(), key=operator.itemgetter(0))
  # Sort by count
  sorted_pair_list = sorted(sorted_pair_list, key=operator.itemgetter(1), reverse=True)
  return sorted_pair_list


def load_json(json_fname):
  colorlog.info("Load %s" % (json_fname))
  with open(json_fname, 'r') as f:
    json_object = json.load(f)
  return json_object


def tokenize(sentences):
  """Tokenize a sentence"""
  if isinstance(sentences, list):
    sentences = ' '.join(sentences)

  # Change separator to space
  sentences = sentences.split(',')

  output = []
  for sentence in sentences:
    sentence = urllib.unquote_plus(sentence)
    sentence = sentence.strip()

    # Remove https
    sentence = _HREF_RE.sub("", sentence)
    sentence = _URL_RE.sub("", sentence)

    # Remove <b></b>
    sentence = re.sub(r"<b>", " ", sentence)
    sentence = re.sub(r"</b>", " ", sentence)

    # Delete square bracket and +
    sentence = re.sub('\[', " ", sentence)
    sentence = re.sub('\]', " ", sentence)
    sentence = re.sub('\+', " ", sentence)
    sentence = re.sub('\)', " ", sentence)
    sentence = re.sub('\(', " ", sentence)
    sentence = re.sub('~', " ", sentence)
    sentence = re.sub('=', " ", sentence)

    # Lowercase
    sentence = sentence.lower()

    # Delete punctuations
    sentence = re.sub(r"([.\*,!?\"@#'|:;)(])", "", sentence)

    # Delte EMOJI
    sentence = EMOTICON.sub(r' ', sentence)

    # Run unidecode
    sentence = unidecode(sentence)
    sentence = NOT_EMOTICON.sub(r' ', sentence)
    sentence = re.sub(r'\\\\', '', sentence)
    sentence = re.sub('\/', '', sentence)
    sentence = re.sub(r'\\', '', sentence)

    # Normalize digit
    sentence = _DIGIT_RE.sub(b"0", sentence)
    sentence = re.sub(r"(?<![a-zA-Z])0(?![a-zA-Z])", r"", sentence)  # remove "-" if there is no preceed or following

    # Incoporate - and -
    sentence = re.sub(r"[\-_]", r"-", sentence)
    sentence = re.sub(r"(?<![a-zA-Z0-9])\-(?![a-zA-Z0-9])", r"", sentence)  # remove "-" if there is no preceed or following

    # Escape unicode
    sentence = sentence.encode('unicode-escape').decode('unicode-escape').encode('ascii', 'ignore').decode('ascii')
    splitted_sentence = sentence.split()
    if len(splitted_sentence) == 2:
      output.append('_'.join(splitted_sentence))
    elif len(splitted_sentence) == 1:
      output.append(splitted_sentence[0])

  return output


def tokenize_all(train_json, test_json):
  """
  Tokenize sentences in raw dataset

  Args:
    train_json, test_json: raw json object
    key: 'caption' or 'tags'
  """

  token_counter = Counter()
  train_tokens = {}
  test_tokens = {}

  # Train data
  for user_id, posts in tqdm(train_json.items(), ncols=70, desc="train data"):
    train_tokens[user_id] = {}
    for post in posts:
      tags = tokenize(post['user tags'])
      post_id = post['page url'].split('/')[-2]
      post_tokens = tags
      train_tokens[user_id][post_id] = post_tokens
      for post_token in post_tokens:
        token_counter[post_token] += 1

  # Test data
  for user_id, posts in tqdm(test_json.items(), ncols=70, desc="test data"):
    test_tokens[user_id] = {}
    for post in posts:
      tags = tokenize(post['user tags'])
      post_id = post['page url'].split('/')[-2]
      post_tokens = tags
      test_tokens[user_id][post_id] = post_tokens


  return token_counter, train_tokens, test_tokens

def tokenize_category(train_json, test_json):
  """
  Tokenize sentences in raw dataset

  Args:
    train_json, test_json: raw json object
    key: 'caption' or 'tags'
  """
  train_tokens = {}
  test_tokens = {}
  
  # Train data
  for user_id, posts in tqdm(train_json.items(), ncols=70, desc="train data"):
    train_tokens[user_id] = {}
    for post in posts:
      tags = post['category']
      post_id = post['page url'].split('/')[-2]
      post_tokens = str(tags)
      post_tokens =post_tokens.replace(" ","")
      post_tokens =post_tokens.strip("[""]")
      post_tokens = post_tokens.replace(",","")
      train_tokens[user_id][post_id] = post_tokens
      

  # Test data
  for user_id, posts in tqdm(test_json.items(), ncols=70, desc="test data"):
    test_tokens[user_id] = {}
    for post in posts:
      tags = post['category']
      post_id = post['page url'].split('/')[-2]
      post_tokens = str(tags)
      post_tokens =post_tokens.replace(" ","")
      post_tokens =post_tokens.strip("[""]")
      post_tokens = post_tokens.replace(",","")
      # print("test_toke, ", post_tokens)
      test_tokens[user_id][post_id] = post_tokens


  return train_tokens, test_tokens


def get_tfidf_words(train_tokens, test_tokens, vocab, rev_vocab):
  colorlog.info("Get tfidf words")
  def _preprocess(all_tokens, rev_vocab):
    counter = np.zeros([len(all_tokens), len(rev_vocab)])
    user_ids = []
    for i, (user_id, posts) in enumerate(
        tqdm(all_tokens.items(), ncols=70, desc="preprocess")
    ):
      user_ids.append(user_id)
      for post_id, tokens in posts.items():
        token_ids = [rev_vocab.get(token, UNK_ID) for token in tokens]
        for token_id in token_ids:
          counter[i, token_id] += 1
    return counter, user_ids

  train_counter, train_user_ids = _preprocess(train_tokens, rev_vocab)
  test_counter, test_user_ids = _preprocess(test_tokens, rev_vocab)

  colorlog.info("Fit and transform train tfidf")
  vectorizer = TfidfTransformer()
  train_tfidf = vectorizer.fit_transform(train_counter).toarray()
  test_tfidf = vectorizer.transform(test_counter).toarray()

  def _extract_tokens(tfidfs, user_ids, vocab):
    user_tokens = {}
    for i, user_id in enumerate(user_ids):
      tfidf = np.argsort(-tfidfs[i])[:CONTEXT_LENGTH]
      weight = np.sort(-tfidfs[i])[:CONTEXT_LENGTH]
      tokens = []
      for j, (index, token_weight) in enumerate(zip(tfidf, weight)):
        token = vocab[index]
        if token_weight < 0.0:
          if index != UNK_ID:
            tokens.append(token)
        else:
          break
      user_tokens[user_id] = tokens
    return user_tokens

  colorlog.info("Extract tokens from tfidf matrix")
  train_user_tokens = _extract_tokens(train_tfidf, train_user_ids, vocab)
  test_user_tokens = _extract_tokens(test_tfidf, test_user_ids, vocab)

  return train_user_tokens, test_user_tokens


def create_vocabulary(counter, fname, vocab_size):
  colorlog.info("Create vocabulary %s" % (fname))
  sorted_tokens = sort_dict(counter)
  vocab = _START_VOCAB + [x[0] for x in sorted_tokens]
  if len(vocab) > vocab_size:
    vocab = vocab[:vocab_size]
  with open(fname, 'w') as f:
    for w in vocab:
      f.write(w + "\n")

  rev_vocab = {}
  for i, token in enumerate(vocab):
    rev_vocab[token] = i

  return vocab, rev_vocab


def save_data(train_data, test_data, output_path, rev_vocab, remove_unk=False):
  """
  Data format:
    numpyfname,contextlength,captionlength,contexttoken1_contexttoken2,wordtoken1_wordtoken2
    e.g. 12345.npy,4,3,445_24_445_232,134_466_234
  """
  def _save_data(all_tokens, all_tfidf, all_category, fname, remove_unk=True):
    all_strings = []
    for user_id, posts in all_tokens.items():
      context_tokenids = map(
          str, [rev_vocab.get(token, UNK_ID) for token in all_tfidf[user_id]]
      )
      context_length = str(len(context_tokenids))
      context_string = '_'.join(context_tokenids)
      for post_id, tokens in posts.items():
        caption_tokenids = map(
            str, [rev_vocab.get(token, UNK_ID) for token in tokens]
        )
        if remove_unk:
          filtered_tokenids = []
          for tokenid in caption_tokenids:
            if tokenid != str(UNK_ID):
              filtered_tokenids.append(tokenid)
            caption_tokenids = filtered_tokenids
        caption_length = str(len(caption_tokenids))
        caption_string = '_'.join(caption_tokenids)
        numpy_string = '%s_%s.npy' % (user_id, post_id)
        category = all_category[user_id][post_id]

        all_string = ','.join([
            numpy_string, context_length, caption_length,
            context_string, caption_string, category
        ])
        all_strings.append((all_string + '\n', len(caption_tokenids)))

    # sort by caption length
    all_strings = sorted(all_strings, key=lambda x: x[1])

    with open(fname, 'w') as f:
      for all_string in all_strings:
        f.write(all_string[0])

  _save_data(
      train_data[0], train_data[1],train_data[2], os.path.join(output_path, "train.txt")
  )
  _save_data(
      test_data[0], test_data[1], test_data[2], os.path.join(output_path, "test.txt"),
      False
  )



def main():
  colorlog.basicConfig(
      filename=None,
      level=logging.INFO,
      format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
      datafmt="%Y-%m-%d %H:%M:%S"
  )

  if not os.path.exists(HASHTAG_OUTPUT_PATH):
    colorlog.info("Create directory %s" % (HASHTAG_OUTPUT_PATH))
    os.makedirs(HASHTAG_OUTPUT_PATH)

  # Load raw data
  hashtag_train_json = load_json(HASHTAG_TRAIN_JSON_FNAME)
  hashtag_test_json = load_json(HASHTAG_TEST_JSON_FNAME)

  # Tokenize all
  hashtag_counter, hashtag_train_tokens, hashtag_test_tokens = tokenize_all(
          hashtag_train_json,
          hashtag_test_json
      )
  hashtag_train_cate, hashtag_test_cate = tokenize_category(
          hashtag_train_json,
          hashtag_test_json
      )

  with open('hashcounter.txt', 'w') as f:
    for key, value in hashtag_counter.most_common():
      f.write("%s : %d\n" % (key, value))

  # Create vocabulary
  hashtag_vocab, hashtag_rev_vocab = create_vocabulary(
      hashtag_counter, HASHTAG_VOCAB_FNAME, HASHTAG_VOCAB_SIZE
  )

  # Get tfidf weighted tokens
  hashtag_train_tfidf_tokens, hashtag_test_tfidf_tokens \
       = get_tfidf_words(
          hashtag_train_tokens,
          hashtag_test_tokens,
          hashtag_vocab,
          hashtag_rev_vocab
      )

  # Save data
  save_data(
      (hashtag_train_tokens, hashtag_train_tfidf_tokens, hashtag_train_cate),
      (hashtag_test_tokens, hashtag_test_tfidf_tokens, hashtag_test_cate),
      HASHTAG_OUTPUT_PATH,
      hashtag_rev_vocab,
      True
  )

if __name__ == '__main__':
  main()
