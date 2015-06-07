#!/bin/bash

# The python script of scoring query with existing
# NMF result.
prog=nmfScore.py

# the list which contains query terms (word).
queryList=qtermlist_1

# THe word map for converting word to its id.
map=wordMap

# The # of latent sementics in NMF result.
dim=90

# Determines how many N-best results to save.
ranknum=7777

for query in `cat $queryList`
do
echo "Scoring $query ..."; \
python $prog $map $dim $query $ranknum;
done
