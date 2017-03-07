#!/usr/bin/env python
#
# Read doxygen xml files to find all members of the dolfin
# name space and generate API doc files per subdirectory of
# dolfin
#
# Written by Tormod Landet, 2017
#
from __future__ import print_function
import sys, os, urllib
import xml.etree.ElementTree as ET


DOXYGEN_XML_DIR = 'doxygen/xml'
API_GEN_DIR = 'generated_rst_files'


def get_single_element(parent, name):
    """
    Helper to get one unique child element
    """
    elems = parent.findall(name)
    assert len(elems) == 1
    return elems[0]


def get_subdir(hpp_file_name):
    """
    Return "subdir" for a path name like
        /path/to/dolfin/subdir/a_header.h
    """
    path_components = hpp_file_name.split(os.sep)
    path_components_rev = path_components[::-1]
    idx = path_components_rev.index('dolfin')
    subdir = path_components_rev[idx - 1]
    return subdir


def get_short_path(hpp_file_name):
    """
    Return "dolfin/subdir/a_header.h" for a path name like
        /path/to/dolfin/subdir/a_header.h
    """
    path_components = hpp_file_name.split(os.sep)
    path_components_rev = path_components[::-1]
    idx = path_components_rev.index('dolfin')
    short_path = path_components_rev[:idx + 1]
    return os.sep.join(short_path[::-1])


def visit_compound(c, xml_file_name):
    """
    Visit xml node describing a class
    Classes have their own xml files
    """
    kind = c.attrib['kind']
    name = get_single_element(c, 'compoundname').text
    hpp_file_name = get_single_element(c, 'location').attrib['file']
    subdir = get_subdir(hpp_file_name)
    short_name = name

    insert_info(subdir, kind, name, short_name, xml_file_name, hpp_file_name)


def visit_namespace_member(m, xml_file_name, namespace):
    """
    Functions and enums are noy described in their own files
    We visit them one by one as elements in a namespace xml file
    """
    kind = m.attrib['kind']
    hpp_file_name = get_single_element(m, 'location').attrib['file']
    name = get_single_element(m, 'name').text
    subdir = get_subdir(hpp_file_name)

    # Make sure we have the required "dolfin::" prefix
    required_prefix = '%s::' % namespace
    if not name.startswith(required_prefix):
        name = required_prefix + name
    short_name = name[len(required_prefix):]

    # Deal with function overloading
    if kind == 'function':
        #name = get_single_element(m, 'definition').text
        argsstring = get_single_element(m, 'argsstring').text
        name += argsstring

    insert_info(subdir, kind, name, short_name, xml_file_name, hpp_file_name)


all_compounds = {}
def insert_info(subdir, kind, name, short_name, xml_file_name, hpp_file_name):
    sd = all_compounds.setdefault(subdir, {})
    kd = sd.setdefault(kind, {})
    kd[name] = (short_name, xml_file_name, hpp_file_name)


def write_rst(subdir):
    kinds = all_compounds[subdir]
    rst_name = os.path.join(API_GEN_DIR, 'api_gen_%s.rst' % subdir)
    print('Generating', rst_name)

    prev_short_name = ''
    with open(rst_name, 'wt') as rst:
        #rst.write('dolfin/%s\n%s' % (subdir, '=' * 80))
        #rst.write('\nDocumentation for C++ code found in dolfin/%s/*.h\n\n' % subdir)
        rst.write('\n\n.. contents::\n\n')
        
        members = [('typedef', 'Type definitions', 'doxygentypedef'),
                   ('enum', 'Enumerations', 'doxygenenum'),
                   ('function', 'Functions', 'doxygenfunction'),
                   ('struct', 'Structures', 'doxygenstruct'),
                   ('variable', 'Variables', 'doxygenvariable'),
                   ('class', 'Classes', 'doxygenclass')]
        
        for kind, kind_name, directive in members:
            if kind in kinds:
                # Write header H2
                rst.write('%s\n%s\n\n' % (kind_name, '-'*70))

                for name, misc in sorted(kinds[kind].items()):
                    short_name, xml_file_name, hpp_file_name = misc
                    fn = get_short_path(hpp_file_name)
                    
                    # Write header H3
                    if short_name != prev_short_name:
                        rst.write('%s\n%s\n\n' % (short_name, '~'*60))
                    prev_short_name = short_name

                    # Info about filename
                    rst.write('C++ documentation for ``%s`` from ``%s``:\n\n' % (short_name, fn))

                    # Breathe autodoc
                    rst.write('.. %s:: %s\n' % (directive, name))
                    rst.write('   :project: dolfin\n')
                    if kind == 'class':
                        rst.write('   :members:\n')
                        rst.write('   :undoc-members:\n\n')
                    else:
                        rst.write('\n')


###############################################################################
# Loop through xml files of compounds and get class definitions
print('Parsing doxygen XML files to make groups of %s/*.rst' % API_GEN_DIR)
xml_files = os.listdir(DOXYGEN_XML_DIR)
errors = []
for file_name in xml_files:
    # Known errors / files we do not care about
    if (file_name in ('namespacestd.xml', 'indexpage.xml')
        or file_name.startswith('group__')):
        continue

    path = os.path.join(DOXYGEN_XML_DIR, file_name)
    root = ET.parse(path).getroot()
    compounds = root.findall('compounddef')
    for c in compounds:
        try:
            visit_compound(c, file_name)
            print('.', end='')
            sys.stdout.flush()
        except Exception:
            errors.append(file_name)

print('\nParsing namespace files')


###############################################################################
# Loop through other elemens in the namespaces
for namespace in ('dolfin', ):
    file_name = 'namespace%s.xml' % namespace
    path = os.path.join(DOXYGEN_XML_DIR, file_name)
    root = ET.parse(path).getroot()
    compound = get_single_element(root, 'compounddef')
    sections = compound.findall('sectiondef')
    for s in sections:
        members = s.findall('memberdef')
        for m in members:
            visit_namespace_member(m, file_name, namespace)
print('Done parsing files')


for file_name in errors:
    print('ERROR: could not parse', file_name)


# Make output directory
if not os.path.isdir(API_GEN_DIR):
    os.mkdir(API_GEN_DIR)


# Generate rst files
for subdir, kinds in sorted(all_compounds.items()):
    if subdir:
        write_rst(subdir)

