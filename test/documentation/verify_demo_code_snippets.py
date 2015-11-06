# Copyright (C) 2010-2014 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Marie E. Rognes (meg@simula.no)
# Modified by Martin Sandve Alnaes, 2014

"""This utility script will find all *.rst files in the source/demo
directory and checks that any code snippets highlighted by the .. code-block::
directive is legal in the sense that it is present in at least one of the
source files (.ufl, .py, .cpp) that is associated with the demo."""

from __future__ import print_function
import sys
import os
from os import chdir, path, getcwd, curdir, pardir, listdir
from sys import stderr, path as sys_path

def verify_blocks(rst_file, source_files, source_dict):
    """Check that any code blocks in the rst file is present in at
    least one of the source files. Returns (False, block) if any block
    is not present, True otherwise."""

    for block_type, source_type in list(source_dict.items()):
        # Extract code blocks from rst file.
        blocks = get_blocks(rst_file, block_type)
        for line, block in blocks:
            sources = [sf for sf in source_files
                       if path.splitext(sf)[-1] == source_type]

            # Check if block is in the list of files of correct type.
            in_source = block_in_source(line, block, sources)
            if not in_source:
                return (False, block)

    return (True, None)

def get_blocks(rst_file, block_type):
    "Extract any code blocks of given type from the rst file."

    blocks = []

    # Open file and read lines.
    f = open(rst_file, "r")
    lines = f.read().split("\n")

    code = False
    block = []
    for e, l in enumerate(lines):
        # Don't tolerate non-space whitespace
        if l.lstrip() != l.lstrip(' '):
            raise RuntimeError("There are tabs or other suspicious whitespace "
                               " in '%s'!" % rst_file)
        # Get start of code block.
        if "code-block::" in l and block_type in l:
            assert not code
            code = True
            block = []
            continue
        # The first line which is not an indented line terminates the code
        # block.
        if code and l and l[0] != " ":
            code = False
            # Join the block that we have and add to list of blocks.
            # Remove any whitespace.
            blocks.append((e, remove_whitespace("\n".join(block))))
        # If code is still True, then the line is part of the code block.
        if code:
            block.append(l)

    # Add block of code if found at the end of the rst file.
    if code:
        blocks.append((e, remove_whitespace("\n".join(block))))

    # Close file and return blocks.
    f.close()
    return blocks

def remove_whitespace(code):
    "Remove blank lines and whitespace in front of lines."
    return "\n".join([" ".join(l.split())\
                      for l in code.split("\n") if l != ""])

def block_in_source(line, block, source_files):
    """Check that the code block is present in at least one of
    the source files."""

    present = False
    code = ""

    # Return fail if no source files are provided
    if not source_files:
        print("\ncode block:\n", block)
        raise RuntimeError("No source file!")

    # Go through each source file and check if any contains code block
    for sf in source_files:
        # Read code and remove whitespace before comparing block and code.
        f = open(sf, "r")
        code = remove_whitespace(f.read())
        f.close()

        # Check whether code block is in code. If so, look no further.
        if block in code:
            return True

    #print "\nError:"
    #print "\ncode line:\n", line
    #print "\ncode block:\n", block
    #print "\nsource_files:\n", source_files
    #print "\nin directory: ", getcwd()
    #print
    #raise RuntimeError("Illegal code block.")
    return False

def main():
    # Make sure we start where this test script is located.
    chdir(sys_path[0])

    # We currently only verify demo code.
    chdir(path.join(pardir, pardir, "demo"))

    # We have C++ and Python versions of the demos.
    directories = ["cpp", "python"]

    # Dictionary of code blocks that has to be checked for each subdirectory
    # including information about file types of the source.
    block_source =  {"cpp":     {"c++": ".cpp", "python": ".ufl"},
                     "python":  {"python": ".py"}
                    }

    # Loop categories/demos/directories

    # Get all demo categories (fem, la. pde, etc.)
    #categories = [d for d in listdir(curdir) if path.isdir(d)]
    # For now only check la/pde
    categories = ["documented"]
    #categories = ["pde"]

    failed = []
    for category in categories:
        chdir(category)
        # Get all demos (Poisson, mixed-Poisson etc.)
        demos = [d for d in listdir(curdir) if path.isdir(d)]

        stderr.write("\nChecking %s demos: %s\n" % (category, str(demos)))
        for demo in demos:
            chdir(demo)
            for directory in directories:
                if not os.path.isdir(directory):
                    continue
                chdir(directory)
                stderr.write("Checking %s: " % path.join(category, demo, directory))
                # Get files in demo directory and sort in rst and source files.
                files = listdir(curdir)
                rst_files = [f for f in files if path.splitext(f)[-1] == ".rst"]
                source_files = [f for f in files if path.splitext(f)[-1] in\
                                  (".py", ".ufl", ".cpp")]

                # If no .rst files are found, that is ok, but suboptimal.
                if (len(rst_files) == 0):
                    stderr.write("*** MISSING ***\n")
                    # FIXME: Enable at some point
                    #failed += [demo]

                # Loop files, check if code blocks are present in source files.
                for rst_file in rst_files:
                    (ok, block) = verify_blocks(rst_file, source_files,
                                                block_source[directory])
                    if not ok:
                        stderr.write("*** FAILED ***\n")
                        stderr.write("\nFailing block:\n\n %s\n\n" % block)
                        failed += [demo]
                    else:
                        stderr.write("OK\n")
                chdir(pardir)
            chdir(pardir)
        chdir(pardir)

    return len(failed)

if __name__ == "__main__":
    sys.exit(main())
