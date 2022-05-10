#!/bin/bash

module load parallel
find $1 -name 'eval_*' | xargs -I {} bash -c "sed -n '/True/p' {} | wc -l" | sort | uniq -c | awk '{ sub(/^[ \t]+/, ""); print }' -
#find $1 -name 'eval_*' -print0 | parallel -0 -I{} bash -c "sed -n '/True/p' {} | wc -l" | sort | uniq -c | awk '{ sub(/^[ \t]+/, ""); print }' -
