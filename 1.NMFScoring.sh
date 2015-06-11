#!/bin/bash

# The python script of scoring query with existing
# NMF result.
prog=nmfScore.py

# the list which contains query terms (word).
#queryList=qtermlist_1
#queryList=qtermlist_2
queryList=qtermlist_3

# THe word map for converting word to its id.
map=wordMap

# Prefix of W,H matrix filename.
prefix=nmf_a

# The # of latent sementics in NMF result.
dim=177

# Determines how many N-best results to save.
ranknum=7777

##outDir=scores

for query in `cat $queryList`
do
echo "Scoring $query ..."; \
python $prog $map $prefix $dim $query $ranknum;
#mv $query_id.txt $outDir
#mv $query_score.txt $outDir
done
