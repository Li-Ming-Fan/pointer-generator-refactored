
import os
import time

import data_utils

import decoding_beam_search
from vocab import STOP_DECODING
import pyrouge_utils

#
def do_train(model, batcher, settings):
    """
    """    
    # train
    #
    model.logger.info("")
    #
    t0 = time.time()
    while True:
        #
        # train
        batch = batcher.get_next_batch()  
        # if batch is None: break
        #
        results_train = model.run_train_one_batch(batch)   # just for train
        global_step = results_train["global_step"]
        #
        if global_step % settings.check_period_batch == 0:
            model.save_ckpt(settings.model_dir, settings.model_name, global_step)
            #
            loss = results_train["loss_optim"]
            print("global_step, loss: %d, %f" % (global_step, loss))
            #
            model.logger.info("global_step, loss: %d, %f" % (global_step, loss))
            # model.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr)
            #
            #
            t1 = time.time()
            # tf.logging.info('seconds for training sect steps: %.3f', t1-t0)
            print('seconds for training sect steps: %.3f' % (t1-t0) )
            t0 = time.time()
            #
    #

#
def do_eval(model, batcher, settings):
    """
    """
    # eval
    #
    model.logger.info("")
    while True:
        #
        # train
        batch = batcher.get_next_batch()  
        # if batch is None: break
        #
        results_eval = model.run_eval_one_batch(batch)   # just for train
        #
        print(results_eval.keys())
        #
    #
    
#
def do_decode(model, batcher, settings):
    """
    """
    vocab = settings.vocab
    #
    # decode
    counter = 0
    while True:
        #
        batch = batcher.get_next_batch()  # 1 example repeated across batch
        #
        if batch is None: # finished decoding dataset in single_pass mode
            assert settings.single_pass, "Dataset exhausted, but we are not in single_pass mode"
            print("Decoder has finished reading dataset for single_pass.")
            print("Output has been saved in %s and %s. Now starting ROUGE eval...",
                  settings.rouge_dir_references, settings.rouge_dir_results)
            # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
            # rouge_log(results_dict, self._decode_dir)
            return
        #
        original_article = batch["original_articles"][0]  # string
        original_abstract = batch["original_abstracts"][0]  # string
        original_abstract_sents = batch["original_abstracts_sents"][0]  # list of strings
        
        article_withunks = data_utils.show_art_oovs(original_article, vocab) # string
        abstract_withunks = data_utils.show_abs_oovs(original_abstract, vocab,
                                                     (batch["art_oovs"][0] if settings.using_pointer_gen else None)) # string
        
        # Run beam search to get best Hypothesis
        best_hyp = decoding_beam_search.run_beam_search(model, batch, vocab, settings)
        
        #
        print("---------------------------------------------------------------------------")
        print_results(article_withunks, abstract_withunks, "")
        #
        
        # Extract the output ids from the hypothesis and convert back to words
        output_ids = [int(t) for t in best_hyp.tokens[1:]]
        decoded_words = data_utils.outputids2words(output_ids, vocab,
                                             (batch["art_oovs"][0] if settings.using_pointer_gen else None))
        
        # Remove the [STOP] token from decoded_words, if necessary
        try:
            first_stop_idx = decoded_words.index(STOP_DECODING) # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:first_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        #    
        decoded_output = ' '.join(decoded_words) # single string
        
        if settings.single_pass:
            write_for_rouge(original_abstract_sents, decoded_words, counter)
            counter += 1 # this is how many examples we've decoded
        else:
            # print_results(article_withunks, abstract_withunks, decoded_output)
            print_results("", "", decoded_output)
        #
        print("---------------------------------------------------------------------------")
        #

#
def print_results(article, abstract, decoded_output):
    """
    """
    # print("---------------------------------------------------------------------------")
    print('ARTICLE: \n%s' % article)
    print('REFERENCE SUMMARY: \n%s' % abstract)
    print('GENERATED SUMMARY: \n%s' % decoded_output)
    # print("---------------------------------------------------------------------------")

#
def write_for_rouge(dir_references, dir_resuts, reference_sents, model_results, example_index):
    """ Write output to file in correct format for eval with pyrouge.
        This is called in single_pass mode.
        
    Args:
        reference_sents: list of strings
        decoded_words: list of strings
        ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(model_results) > 0:
        try:
            first_period_idx = model_results.index(".")
        except ValueError: # there is text remaining that doesn't end in "."
            first_period_idx = len(model_results)
        #
        sent = model_results[:first_period_idx+1] # sentence up to and including the period
        model_results = model_results[first_period_idx+1:] # everything remaining
        #
        decoded_sents.append(' '.join(sent))
        #

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [pyrouge_utils.make_html_safe(w) for w in decoded_sents]
    reference_sents = [pyrouge_utils.make_html_safe(w) for w in reference_sents]

    # Write to file
    reference_file = os.path.join(dir_references, "%06d_reference.txt" % example_index)
    result_file = os.path.join(dir_resuts, "%06d_result.txt" % example_index)

    with open(reference_file, "w", encoding="utf-8") as f:
        for idx,sent in enumerate(reference_sents):
            f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(result_file, "w", encoding="utf-8") as f:
        for idx,sent in enumerate(decoded_sents):
            f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")
    #
    