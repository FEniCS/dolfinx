"""
Produce docstrings for SWIG.

We run doxygen and parse the resulting XML to generate ReStucturedText
"""
from __future__ import print_function
from codesnippets import copyright_statement
import os
import sys
import subprocess
from dummy_docstrings import generate_dummy_docstrings


PROBLEMATIC_DOXYGEN_VERSIONS = ['1.8.13']


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


def get_required_swig_files(dolfin_dir, swig_dir):
    """
    Return the names of the docstring.i files we are required to generate
    """
    swig_files = []
    dolfin_dir = os.path.join(dolfin_dir, 'dolfin')
    for subdir in os.listdir(dolfin_dir):
        path = os.path.join(dolfin_dir, subdir)
        if not os.path.isdir(path) or subdir == 'swig':
            continue
        swig_files.append(os.path.join(swig_dir, subdir, 'docstrings.i'))
    return swig_files


def generate_docstrings(top_destdir):
    """
    Generate docstring files for each module
    """
    doxyver = get_doxygen_version()
    
    if doxyver is None:
        print('--------------------------------------------')
        print('WARNING: Missing doxygen, producing dummy docstrings')
        generate_dummy_docstrings(top_destdir, reason='Missing doxygen')
        return
    
    # Get top DOLFIN directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dolfin_dir = os.path.abspath(os.path.join(script_dir, os.pardir,
                                              os.pardir))
    
    print('--------------------------------------------')
    print('Running doxygen to read docstrings from C++:')
    print('Doxygen version:', doxyver)
    
    dummy_reason = ''
    if str(doxyver) in PROBLEMATIC_DOXYGEN_VERSIONS:
        print('WARNING: this doxygen version has known problems with dolfin')
        dummy_reason = 'Buggy doxygen version %s. ' % doxyver
    
    sys.stdout.flush() # doxygen writes to stderr and mangles output order
    
    # Get doc directory and run doxygen there
    doc_dir = os.path.join(dolfin_dir, 'doc')
    xml_dir = os.path.join(dolfin_dir, 'doc', 'doxygen', 'xml')
    try:
        subprocess.call(['doxygen'], cwd=doc_dir)
    except OSError as e:
        print('ERROR: could not run doxygen:', e)
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
    
    # Remove destination files so that we can check if we succeeded
    required_swig_files = get_required_swig_files(dolfin_dir, swig_dir)
    for file_name in required_swig_files:
        if os.path.isfile(file_name):
            os.remove(file_name)
    
    print('--------------------------------------------')
    print('Generating python docstrings in directory')
    print('  swig_dir = %s' % swig_dir)
    
    # Extract documentation and generate docstrings
    if os.path.isdir(xml_dir):
        parse_doxygen_xml_and_generate_rst_and_swig(xml_dir=xml_dir,
                                                    api_gen_dir=None,
                                                    swig_dir=swig_dir,
                                                    swig_file_name="docstrings.i",
                                                    swig_header=copyright_info)
    else:
        # Doxygen crashed before generating any XML
        dummy_reason += 'Doxygen did not generate any XML files. '
    
    # Check that all files have been generated
    # Doxygen may have crashed halfway through generating the XML
    all_files_generated = True
    for file_name in required_swig_files:
        if not os.path.isfile(file_name):
            all_files_generated = False
            print('ERROR: missing file %s' % file_name)
    
    # Generate dummy files if something went wrong
    # Possible errors can be problems with the XML output from doxygen. As 
    # an example doxygen version 1.8.13 segfaults after producing a few XML
    # files while versions 1.8.12 and 1.8.14 works fine
    if not all_files_generated:
        print('--------------------------------------------')
        print('WARNING: Something went wrong, producing dummy docstrings')
        dummy_reason += 'Missing generated files. Did doxygen crash?'
        generate_dummy_docstrings(top_destdir, reason=dummy_reason)
    
    print('DONE generating docstrings')
    print('--------------------------------------------')


if __name__ == "__main__":
    if len(sys.argv) not in (1, 2):
        raise RuntimeError("expected 0 or 1 arguments")
    dest_dir = sys.argv[1] if len(sys.argv) > 2 else ""
    generate_docstrings(dest_dir)
