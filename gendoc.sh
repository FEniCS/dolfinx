#!/bin/sh

# This script calls doc++ with the proper flags to
# create some documentation.
#
# Note: If files are not already docified you need to run
#
#    docify *.C *.h *.hh
#
# on the files.

SRC_DIR=src/fem
HTML_DIR=doc/html
TEX_FILE=doc/tex/doc.tex

FILES=`find $SRC_DIR/*.hh`

# Docify files
#for f in $FILES; do
#  rm -f gendoc.tmp
#  docify $f > gendoc.tmp
#  mv gendoc.tmp $f
#done

# Generate the documentaion with doc++
doc++ -M -S -b -f -d $HTML_DIR $FILES
doc++ -S -b -f --tex --output $TEX_FILE $FILES
