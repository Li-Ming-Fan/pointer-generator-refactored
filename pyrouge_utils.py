# -*- coding: utf-8 -*-

import logging
import pyrouge
import json


#
default_rouge_score = {'rouge_1_recall': 0.0, 'rouge_1_recall_cb': 0.0, 'rouge_1_recall_ce': 0.0,
               'rouge_1_precision': 0.0, 'rouge_1_precision_cb': 0.0, 'rouge_1_precision_ce': 0.0,
               'rouge_1_f_score': 0.0, 'rouge_1_f_score_cb': 0.0, 'rouge_1_f_score_ce': 0.0,
               'rouge_2_recall': 0.0, 'rouge_2_recall_cb': 0.0, 'rouge_2_recall_ce': 0.0,
               'rouge_2_precision': 0.0, 'rouge_2_precision_cb': 0.0, 'rouge_2_precision_ce': 0.0,
               'rouge_2_f_score': 0.0, 'rouge_2_f_score_cb': 0.0, 'rouge_2_f_score_ce': 0.0,
               'rouge_3_recall': 0.0, 'rouge_3_recall_cb': 0.0, 'rouge_3_recall_ce': 0.0,
               'rouge_3_precision': 0.0, 'rouge_3_precision_cb': 0.0, 'rouge_3_precision_ce': 0.6,
               'rouge_3_f_score': 0.0, 'rouge_3_f_score_cb': 0.0, 'rouge_3_f_score_ce': 0.0,
               'rouge_4_recall': 0.0, 'rouge_4_recall_cb': 0.0, 'rouge_4_recall_ce': 0.0,
               'rouge_4_precision': 0.0, 'rouge_4_precision_cb': 0.0, 'rouge_4_precision_ce': 0.0,
               'rouge_4_f_score': 0.0, 'rouge_4_f_score_cb': 0.0, 'rouge_4_f_score_ce': 0.0,
               'rouge_l_recall': 0.0, 'rouge_l_recall_cb': 0.0, 'rouge_l_recall_ce': 0.0,
               'rouge_l_precision': 0.0, 'rouge_l_precision_cb': 0.0, 'rouge_l_precision_ce': 0.0,
               'rouge_l_f_score': 0.0, 'rouge_l_f_score_cb': 0.0, 'rouge_l_f_score_ce': 0.0,
               'rouge_w_1.2_recall': 0.0, 'rouge_w_1.2_recall_cb': 0.0, 'rouge_w_1.2_recall_ce': 0.0, 
               'rouge_w_1.2_precision': 0.0, 'rouge_w_1.2_precision_cb': 0.0, 'rouge_w_1.2_precision_ce': 0.0,
               'rouge_w_1.2_f_score': 0.0, 'rouge_w_1.2_f_score_cb': 0.0, 'rouge_w_1.2_f_score_ce': 0.0,
               'rouge_s*_recall': 0.0, 'rouge_s*_recall_cb': 0.0, 'rouge_s*_recall_ce': 0.0, 
               'rouge_s*_precision': 0.0, 'rouge_s*_precision_cb': 0.0, 'rouge_s*_precision_ce': 0.0,
               'rouge_s*_f_score': 0.0, 'rouge_s*_f_score_cb': 0.0, 'rouge_s*_f_score_ce': 0.0,
               'rouge_su*_recall': 0.0, 'rouge_su*_recall_cb': 0.0, 'rouge_su*_recall_ce': 0.0,
               'rouge_su*_precision': 0.0, 'rouge_su*_precision_cb': 0.0, 'rouge_su*_precision_ce': 0.0,
               'rouge_su*_f_score': 0.0, 'rouge_su*_f_score_cb': 0.0, 'rouge_su*_f_score_ce': 0.0}
#
        

#
def make_html_safe(s):
    """ Replace any angled brackets in string s to avoid interfering with HTML attention visualizer.
    
        pyrouge calls a perl script that puts the data into HTML files.
        Therefore we need to make our output HTML safe.
    """
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def run_pyrouge_eval(system_dir, model_dir):
    """
    """
    rg = pyrouge.Rouge155()
    rg.system_dir = system_dir
    rg.model_dir = model_dir
    rg.system_filename_pattern = '(\d+)_reference.txt'
    rg.model_filename_pattern = '#ID#_result.txt'
    #
    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
    output = rg.convert_and_evaluate()
    # print(output)
    output_dict = rg.output_to_dict(output)
    # print(output_dict)
    return output_dict, output


def write_rouge_results(results_dict, file_path):
    """
    """
    results_filtered = {}
    results_filtered["rouge_1_recall"] = results_dict["rouge_1_recall"]
    results_filtered["rouge_1_precision"] = results_dict["rouge_1_precision"]
    results_filtered["rouge_1_f_score"] = results_dict["rouge_1_f_score"]
    #
    results_filtered["rouge_2_recall"] = results_dict["rouge_2_recall"]
    results_filtered["rouge_2_precision"] = results_dict["rouge_2_precision"]
    results_filtered["rouge_2_f_score"] = results_dict["rouge_2_f_score"]
    #
    results_filtered["rouge_4_recall"] = results_dict["rouge_4_recall"]
    results_filtered["rouge_4_precision"] = results_dict["rouge_4_precision"]
    results_filtered["rouge_4_f_score"] = results_dict["rouge_4_f_score"]
    #
    results_filtered["rouge_l_recall"] = results_dict["rouge_l_recall"]
    results_filtered["rouge_l_precision"] = results_dict["rouge_l_precision"]
    results_filtered["rouge_l_f_score"] = results_dict["rouge_l_f_score"]
    #
    with open(file_path, "w") as fp:
        json.dump(results_filtered, fp)

