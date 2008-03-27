#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2006 Simula Research Laboratory
# Author: Ola Skavhaug, Åsmund Ødegård

from os import system
import os, sys
from os.path import sep, join

from commonPkgConfigUtils import *

def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
    return True

# Generate a pkg-config file for dolfin, put it in the given 
# directory, if no directory is given, a suitable location is found 
# using the functionality from commonPkgConfigUtils.
# Nothing in kwargs is used, it is included to ensure a consitent 
# interface for the generatePkgConf function.
def generatePkgConf(directory=suitablePkgConfDir(),**kwargs):
    
    print """\nError:\n\nDOLFIN now supplies its own pkg-config file, which we are
unable to find! Either copy that file to a directory search by pkg-config, or 
add the directory where DOLFIN installs the file to your PKG_CONFIG_PATH 
environment variable. The directory in question is $prefix/lib/pkgconfig, where 
$prefix is the install prefix for DOLFIN\n\n"""

    sys.exit(1)

if __name__ == "__main__":
    generatePkgConf(directory=".")
