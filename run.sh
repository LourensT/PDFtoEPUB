#!/usr/bin/env bash
# replace .pdf with .md in first argument
fpmd=$(echo $1 | sed 's/.pdf/.md/g')
# replace .pdf with .epub in first argument
ofepub=$(echo $1 | sed 's/.pdf/.epub/g')
# print the file paths
echo converting $1 to markdown $fpmd
# activate python venv
source .venv/bin/activate
# set environment variables from .env
export $(cat .env | xargs)
# run conversion to markdown
python main.py $1 $fpmd
# deactivate python venv
deactivate
# convert markdown to pdf
echo converting markdown $fpmd to epub: $ofepub
# convert markdown to epub
pandoc $fpmd -o $ofepub