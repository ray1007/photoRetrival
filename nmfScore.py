#!/usr/bin/python

from sys import argv
from collections import OrderedDict
import os.path
import pdb

import numpy as np

def print_usage(msg):
    usage = (
        "> python nmfScore.py <map_file> <fname_prefix> <model_dim> <query_text> <N-best>\n"
        "\n"
        "  <map_file>    (str): filename of word-id table.\n"
        "  <fname_prefix>(str): filename prefix.\n"
        "  <model_dim>   (int): dimension of d. Need to make\n"
        "                      sure 'nmf_W_<model_dim>' &\n"
        "                      'nmf_H_<model_dim>' exists.\n"
        "  <query_text>  (str): query text string.\n"
        "  <N-best>      (int): output N-best id & score.\n"
    )
    print('\n'.join([msg,usage]))
    return

def parse_argv():
    # parse input arguments.
    if len(argv) != 6: print_usage("Needs 5 arguments."); quit()
    map_file = argv[1]
    if not os.path.isfile(map_file): 
        print_usage("Invalid map file.");
        quit()

    prefix = argv[2]

    if not argv[3].isdigit(): print_usage("Model dim not int."); quit()
    d = int(argv[3])
    if not os.path.isfile('{0}_W_{1}'.format(prefix,d)):
        print_usage('{0}_W_{1} does not exist.'.format(prefix,d)); quit()
    if not os.path.isfile('{0}_H_{1}'.format(prefix,d)):
        print_usage('{0}_H_{1} does not exist.'.format(prefix,d)); quit()

    query = argv[4]
    
    if not argv[5].isdigit(): print_usage("<N-best> not int."); quit()
    n_best = int(argv[5])
    return (map_file, prefix, d, query, n_best)

def nmf_scoring(W, H_col, n_best):
    scoreD = {} # id:score
    for i in xrange(W.shape[0]):
        #pdb.set_trace()
        s = np.dot(W[i], H_col)
        scoreD[i] = s
    scoreD = OrderedDict(sorted(scoreD.items(), key=lambda t: t[1], \
                         reverse=True))
    count = 0
    id_list = []
    score_list = []
    for k,v in scoreD.iteritems():
        if count == n_best: break;
        id_list.append(k)
        score_list.append(np.log(v))
        count+=1
    return (id_list, score_list)

def write_file(id_list, score_list):
    with open('{0}_id.txt'.format(query), 'w') as f:
        for i in id_list:
            f.write('{0}\n'.format(i))
    with open('{0}_score.txt'.format(query), 'w') as f:
        for s in score_list:
            f.write('{0}\n'.format(s))
    return

N = 7777
M = 61509

if __name__ == "__main__":
    (map_file, prefix, d, query, n_best) = parse_argv()

    # build word map.
    word_map = {}
    with open(map_file, 'r') as f:
        for line in f:
            [i,w] = line.split()
            word_map[w]=int(i)

    W = np.memmap('{0}_W_{1}'.format(prefix,d), dtype="float32", \
                  mode='r', shape=(N, d))
    H = np.memmap('{0}_H_{1}'.format(prefix,d), dtype="float32", \
                  mode='r', shape=(d, M))
    query_id = word_map.get(query, -1)
    if query_id == -1: print("OOV."); quit()
    #pdb.set_trace()
    (id_list, score_list) = nmf_scoring(W, H[:,query_id-1], n_best)
    write_file(id_list, score_list)
