#!/bin/sh

# Copyright (C) 2015 Martin Alnæs, Johannes Ring
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This script redirects output from a command to output.MYRANK

# See: http://docs.roguewave.com/threadspotter/2012.1/linux/manual_html/apas03.html
if [ ! -z $OMPI_COMM_WORLD_RANK ]; then
    MYRANK=$OMPI_COMM_WORLD_RANK
elif [ ! -z $PMI_RANK ]; then
    MYRANK=$PMI_RANK
else
    MYRANK=`python -c "import dolfin;c=dolfin.mpi_comm_world();r=dolfin.MPI.rank(c);print(r);dolfin.MPI.barrier(c)"`
fi

OUT=output.$MYRANK
echo My rank is $MYRANK, writing to $OUT
if [ $MYRANK -eq 0 ]; then
  $@ | tee $OUT
else
  $@ > $OUT
fi
