# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import glob
import random
import struct
from tensorflow.core.example import example_pb2

import numpy as np

#
from vocab import SENTENCE_START, SENTENCE_END
from vocab import PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING

#
def article2ids(article_words, vocab):
    """ Map the article words to their ids. Also return a list of OOVs in the article.
    
      Args:
        article_words: list of words (strings)
        vocab: Vocabulary object
    
      Returns:
        ids:
          A list of word ids (integers); OOVs are represented by their temporary article OOV number.
          If the vocabulary size is 50k and the article has 3 OOVs,
          then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs:
          A list of the OOV words in the article (strings),
          in the order corresponding to their temporary article OOV numbers.
    """
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            #
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    #
    return ids, oovs

def abstract2ids(abstract_words, vocab, article_oovs):
    """ Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

      Args:
        abstract_words: list of words (strings)
        vocab: Vocabulary object
        article_oovs: list of in-article OOV words (strings),
        in the order corresponding to their temporary article OOV numbers
    
      Returns:
        ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers.
        Out-of-article OOV words are mapped to the UNK token id.
    """
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    #
    return ids

def outputids2words(id_list, vocab, article_oovs):
    """ Maps output ids to words, 
        including mapping in-article OOVs from their temporary ids to the original OOV string
        (applicable in pointer-generator mode).

      Args:
        id_list: list of ids (integers)
        vocab: Vocabulary object
        article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids
        (that have been assigned in pointer-generator mode), or None (in baseline mode)
    
      Returns:
        words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            error_info = "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            assert article_oovs is not None, error_info
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        #
        words.append(w)
        #
        
    return words

def abstract2sents(abstract):
    """ Splits abstract text from datafile into list of sentences.
    
    Args:
        abstract: string containing <s> and </s> tags for starts and ends of sentences
    
    Returns:
        sents: List of sentence strings (no tags)
    """
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e: # no more sentences
            return sents

def show_art_oovs(article, vocab):
    """ Returns the article string, highlighting the OOVs by placing __underscores__ around them
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    
    return out_str

def show_abs_oovs(abstract, vocab, article_oovs):
    """ Returns the abstract string, highlighting the article OOVs with __underscores__.
    If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.
    
    Args:
        abstract: string
        vocab: Vocabulary object
        article_oovs: list of words (strings), or None (in baseline mode)
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token: # w is oov
            if article_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    #        
    out_str = ' '.join(new_words)
    return out_str


#
class Example(object):
    """ Class representing a train/val/test example for text summarization.
    """
    def __init__(self, article, abstract_sentences, vocab, settings):
        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        Args:
          article: source text; a string. each token is separated by a single space.
          abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
          vocab: Vocabulary object
          settings: hyperparameters
        """
        self.settings = settings

        # Get ids of special tokens
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)
    
        # Process the article
        article_words = article.split()
        if len(article_words) > settings.max_enc_steps:
            article_words = article_words[:settings.max_enc_steps]
        #
        self.enc_len = len(article_words)
        self.enc_input = [vocab.word2id(w) for w in article_words]
        
        # Process the abstract
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words]
        
        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, settings.max_dec_steps,
                                                                 start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)
    
        # If using pointer-generator mode, we need to store some extra info
        if settings.using_pointer_gen:
          # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
          # also store the in-article OOVs words themselves
          self.enc_input_extend_vocab, self.article_oovs = article2ids(article_words, vocab)
    
          # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
          abs_ids_extend_vocab = abstract2ids(abstract_words, vocab, self.article_oovs)
    
          # Overwrite decoder target sequence so it uses the temp article OOV ids
          _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, settings.max_dec_steps,
                                                      start_decoding, stop_decoding)
    
        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences
        
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens,
           return the input sequence for the decoder,
           and the target sequence which we will use to calculate loss.
           The sequence will be truncated if it is longer than max_len.
           The input sequence must start with the start_id and the target sequence must end with the stop_id
           (but not if it's been truncated).
    
        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer
    
        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
            inp = inp[:max_len]
            target = target[:max_len] # no end_token
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)
        
        return inp, target
    
    def pad_decoder_inp_targ(self, max_len, pad_id):
        """ Pad decoder input and target sequences with pad_id up to max_len.
        """
        while len(self.dec_input) < max_len:
          self.dec_input.append(pad_id)
        while len(self.target) < max_len:
          self.target.append(pad_id)
          
    def pad_encoder_input(self, max_len, pad_id):
        """ Pad the encoder input sequence with pad_id up to max_len.
        """
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.settings.using_pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

class Batch(object):
    """ Class representing a minibatch of train/val/test examples for text summarization.
    """
    def __init__(self, example_list, settings, vocab):
        """Turns the example_list into a Batch object.
    
        Args:
           example_list: List of Example objects
           settings: hyperparameters
           vocab: Vocabulary object
        """
        self.pad_id = vocab.word2id(PAD_TOKEN) # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list, settings) # initialize the input to the encoder
        self.init_decoder_seq(example_list, settings) # initialize the input and targets for the decoder
        self.store_orig_strings(example_list) # store the original strings
        
    def init_encoder_seq(self, example_list, settings):
        """Initializes the following:
            self.enc_batch:
              numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
            self.enc_lens:
              numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
            self.enc_padding_mask:
              numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.
    
          If settings.pointer_gen, additionally initializes the following:
            self.max_art_oovs:
              maximum number of in-article OOVs in the batch
            self.art_oovs:
              list of list of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab:
              Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
    
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
    
        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((settings.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((settings.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((settings.batch_size, max_enc_seq_len), dtype=np.float32)
    
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1
    
        # For pointer-generator mode, need to store some extra info
        if settings.using_pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((settings.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]
                
    def init_decoder_seq(self, example_list, settings):
        """Initializes the following:
            self.dec_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
            self.target_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
            self.dec_padding_mask:
              numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
            """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(settings.max_dec_steps, self.pad_id)
    
        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((settings.batch_size, settings.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((settings.batch_size, settings.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((settings.batch_size, settings.max_dec_steps), dtype=np.float32)
    
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1
            
    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_articles = [ex.original_article for ex in example_list] # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


#
# load data
def example_generator(data_path, single_pass):
    """ Generates tf.Examples from data files.
    
        Binary data format: <length><blob>. <length> represents the byte size
        of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
        the tokenized article text and summary.
    
      Args:
        data_path:
          Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
        single_pass:
          Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.
    
      Yields:
        Deserialized tf.Example.
    """
    while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
        #
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                e = example_pb2.Example.FromString(example_str)
                #
                try:
                    article_text = e.features.feature['article'].bytes_list.value[0].decode() # the article text was saved under the key 'article' in the data files
                    abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode() # the abstract text was saved under the key 'abstract' in the data files
                except ValueError:
                    print('Failed to get article or abstract from example')
                    continue
                if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
                    print('Found an example with empty article text. Skipping it.')
                else:
                    # print(abstract_text)
                    # print()
                    yield (article_text, abstract_text)
                #
        #
        if single_pass:
            print("example_generator completed reading all datafiles. No more data.")
            break
        #
      
#
def do_batch_std(list_examples_raw, settings):
    """
    """
    list_examples = []
    for article, abstract in list_examples_raw:
        #
        # Use the <s> and </s> tags in abstract to get a list of sentences.
        abstract_sentences = [sent.strip() for sent in abstract2sents(abstract)]
        #
        example = Example(article, abstract_sentences, settings.vocab, settings)
        list_examples.append(example)
        #
    #
    # print()
    # print(len(list_examples))
    #
    batch = Batch(list_examples, settings, settings.vocab)
    return batch
