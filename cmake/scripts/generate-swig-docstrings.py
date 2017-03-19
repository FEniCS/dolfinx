"""
Produce docstrings for SWIG.

We run doxygen and parse the resulting XML to generate ReStucturedText
"""
from __future__ import print_function
from codesnippets import copyright_statement
import os
import sys
import subprocess


def generate_docstrings(top_destdir):
    """
    Generate docstring files for each module
    """
    # Get top DOLFIN directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dolfin_dir = os.path.abspath(os.path.join(script_dir, os.pardir,
                                              os.pardir))

    # Get doc directory and run doxygen there
    doc_dir = os.path.join(dolfin_dir, 'doc')
    xml_dir = os.path.join(dolfin_dir, 'doc', 'doxygen', 'xml')
    subprocess.call('doxygen', shell=True, cwd=doc_dir)

    top_destdir = top_destdir or dolfin_dir
    abs_destdir = top_destdir if os.path.isabs(top_destdir) else os.path.join(dolfin_dir, top_destdir)

    if not os.path.isdir(abs_destdir):
        raise RuntimeError("%s is not a directory." % abs_destdir)

    # Directory with swig files
    swig_dir = os.path.join(abs_destdir, "dolfin", "swig")

    # Get copyright form
    copyright_form_swig = dict(comment=r"//", holder="Kristian B. Oelgaard, Tormod Landet")
    copyright_info = copyright_statement % copyright_form_swig

    # Get the doxygen parser and docstring formater
    sys.path.insert(0, doc_dir)
    from generate_api_rst import parse_doxygen_xml_and_generate_rst_and_swig

    # Extract documentation and generate docstrings
    parse_doxygen_xml_and_generate_rst_and_swig(xml_dir=xml_dir,
                                                api_gen_dir=None,
                                                swig_dir=swig_dir,
                                                swig_file_name="docstrings.i",
                                                swig_header=copyright_info)


if __name__ == "__main__":
    if len(sys.argv) not in (1, 2):
        raise RuntimeError("expected 0 or 1 arguments")
    dest_dir = sys.argv[1] if len(sys.argv) > 2 else ""
    generate_docstrings(dest_dir)
