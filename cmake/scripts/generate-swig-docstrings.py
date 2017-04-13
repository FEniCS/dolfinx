"""
Produce docstrings for SWIG.

We run doxygen and parse the resulting XML to generate ReStucturedText
"""
from __future__ import print_function
from codesnippets import copyright_statement
import os
import sys
import subprocess


def get_doxygen_version():
    """
    Test for presence of doxygen
    
    Returns None if doxygen is not present, else a string containing
    the doxygen version
    """
    try:
        ver = subprocess.check_output(['doxygen', '--version'])
    except Exception:
        return None
    return ver.decode('utf8', 'replace').strip()


def generate_docstrings(top_destdir):
    """
    Generate docstring files for each module
    """
    doxyver = get_doxygen_version()
    
    if doxyver is None:
        print('--------------------------------------------')
        print('WARNING: Missing doxygen, producing dummy docstrings')
        from dummy_docstrings import generate_dummy_docstrings
        generate_dummy_docstrings(top_destdir)
        return
    
    # Get top DOLFIN directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dolfin_dir = os.path.abspath(os.path.join(script_dir, os.pardir,
                                              os.pardir))
    
    print('--------------------------------------------')
    print('Running doxygen to read docstrings from C++:')
    print('Doxygen version:', doxyver)
    sys.stdout.flush() # doxygen writes to stderr and mangles output order

    # Get doc directory and run doxygen there
    doc_dir = os.path.join(dolfin_dir, 'doc')
    xml_dir = os.path.join(dolfin_dir, 'doc', 'doxygen', 'xml')
    allow_empty_xml = False
    try:
        subprocess.call(['doxygen'], cwd=doc_dir)
    except OSError as e:
        print('ERROR: could not run doxygen:', e)
        allow_empty_xml = True
    print('DONE parsing C++ with doxygen')

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
    
    print('--------------------------------------------')
    print('Generating python docstrings in directory')
    print('  swig_dir = %s' % swig_dir)

    # Extract documentation and generate docstrings
    parse_doxygen_xml_and_generate_rst_and_swig(xml_dir=xml_dir,
                                                api_gen_dir=None,
                                                swig_dir=swig_dir,
                                                swig_file_name="docstrings.i",
                                                swig_header=copyright_info,
                                                allow_empty_xml=allow_empty_xml)
    
    print('DONE generating docstrings')
    print('--------------------------------------------')


if __name__ == "__main__":
    if len(sys.argv) not in (1, 2):
        raise RuntimeError("expected 0 or 1 arguments")
    dest_dir = sys.argv[1] if len(sys.argv) > 2 else ""
    generate_docstrings(dest_dir)
