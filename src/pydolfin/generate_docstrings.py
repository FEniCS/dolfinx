#!/usr/bin/env python

__author__ = "Johan Hake & Prabhu Ramachandran"
__license__ = "bsd style"

import os

from xml.dom import minidom
import re
import textwrap
import sys
import types

def my_open_read(source):
    if hasattr(source, "read"):
        return source
    else:
        return open(source)

def my_open_write(dest):
    if hasattr(dest, "write"):
        return dest
    else:
        return open(dest, 'w')


class Doxy2SWIG:    
    """Converts Doxygen generated XML files into a file containing
    docstrings that can be used by SWIG-1.3.x that have support for
    feature("docstring").  Once the data is parsed it is stored in
    self.pieces.

    """    
    __author__ = "Prabhu Ramachandran"
    __license__ = "BSD style"
    # This code is implemented using Mark Pilgrim's code as a guideline:
    #   http://www.faqs.org/docs/diveintopython/kgp_divein.html

    def __init__(self, src):
        """Initialize the instance given a source object (file or
        filename).

        """
        f = my_open_read(src)
        self.my_dir = os.path.dirname(f.name)
        self.xmldoc = minidom.parse(f).documentElement
        f.close()

        self.pieces = []
        self.pieces.append('\n// File: %s\n'%\
                           os.path.basename(f.name))

        self.space_re = re.compile(r'\s+')
        self.lead_spc = re.compile(r'^(%feature\S+\s+\S+\s*?)"\s+(\S)')
        self.multi = 0
        self.ignores = ('inheritancegraph', 'param', 'listofallmembers',
                        'innerclass', 'name', 'declname', 'incdepgraph',
                        'invincdepgraph', 'programlisting', 'type',
                        'references', 'referencedby', 'location',
                        'collaborationgraph', 'reimplements',
                        'reimplementedby', 'derivedcompoundref',
                        'basecompoundref', 'argsstring')
        #self.generics = []
        self.include_function_definition = False
        
    def generate(self):
        """Parses the file set in the initialization.  The resulting
        data is stored in `self.pieces`.

        """
        self.parse(self.xmldoc)
    
    def parse(self, node):
        """Parse a given node.  This function in turn calls the
        `parse_<nodeType>` functions which handle the respective
        nodes.

        """
        pm = getattr(self, "parse_%s"%node.__class__.__name__)
        pm(node)

    def parse_Document(self, node):
        self.parse(node.documentElement)

    def parse_Text(self, node):
        txt = node.data
        txt = txt.replace('\\', r'\\\\')
        txt = txt.replace('"', r'\"')
        # ignore pure whitespace
        m = self.space_re.match(txt)
        if m and len(m.group()) == len(txt):
            pass
        else:
            self.add_text(textwrap.fill(txt))

    def parse_Element(self, node):
        """Parse an `ELEMENT_NODE`.  This calls specific
        `do_<tagName>` handers for different elements.  If no handler
        is available the `generic_parse` method is called.  All
        tagNames specified in `self.ignores` are simply ignored.
        
        """
        name = node.tagName
        ignores = self.ignores
        if name in ignores:
            return
        attr = "do_%s" % name
        if hasattr(self, attr):
            handlerMethod = getattr(self, attr)
            handlerMethod(node)
        else:
            self.generic_parse(node)
            #if name not in self.generics: self.generics.append(name)

    def parse_Comment(self, node):
        return

    def add_text(self, value):
        """Adds text corresponding to `value` into `self.pieces`."""
        if type(value) in (types.ListType, types.TupleType):
            self.pieces.extend(value)
        else:
            self.pieces.append(value)

    def get_specific_nodes(self, node, names):
        """Given a node and a sequence of strings in `names`, return a
        dictionary containing the names as keys and child
        `ELEMENT_NODEs`, that have a `tagName` equal to the name.

        """
        nodes = [(x.tagName, x) for x in node.childNodes \
                 if x.nodeType == x.ELEMENT_NODE and \
                 x.tagName in names]
        return dict(nodes)

    def generic_parse(self, node, pad=0):
        """A Generic parser for arbitrary tags in a node.

        Parameters:

         - node:  A node in the DOM.
         - pad: `int` (default: 0)

           If 0 the node data is not padded with newlines.  If 1 it
           appends a newline after parsing the childNodes.  If 2 it
           pads before and after the nodes are processed.  Defaults to
           0.

        """
        npiece = 0
        if pad:
            npiece = len(self.pieces)
            if pad == 2:
                self.add_text('\n')                
        for n in node.childNodes:
            self.parse(n)
        if pad:
            if len(self.pieces) > npiece:
                self.add_text('\n')

    def space_parse(self, node):
        self.add_text(' ')
        self.generic_parse(node)

    do_ref = space_parse
    do_emphasis = space_parse
    do_bold = space_parse
    do_computeroutput = space_parse
    do_formula = space_parse

    def do_compoundname(self, node):
        self.add_text('\n\n')
        data = node.firstChild.data
        self.add_text('%%feature("docstring") %s "\n'%data)

    def do_compounddef(self, node):
        kind = node.attributes['kind'].value
        if kind in ('class', 'struct'):
            prot = node.attributes['prot'].value
            if prot <> 'public':
                return
            names = ('compoundname', 'briefdescription',
                     'detaileddescription', 'includes')
            first = self.get_specific_nodes(node, names)
            for n in names:
                if first.has_key(n):
                    self.parse(first[n])
            self.add_text(['";','\n'])
            for n in node.childNodes:
                if n not in first.values():
                    self.parse(n)
        elif kind in ('file', 'namespace'):
            nodes = node.getElementsByTagName('sectiondef')
            for n in nodes:
                self.parse(n)

    def do_includes(self, node):
        self.add_text('C++ includes: ')
        self.generic_parse(node, pad=1)

    def do_parameterlist(self, node):
        self.add_text(['\n', '\n', 'Parameters:', '\n'])
        self.generic_parse(node, pad=1)

    def do_para(self, node):
        self.add_text('\n')
        self.generic_parse(node, pad=1)

    def do_parametername(self, node):
        self.add_text('\n')
        self.add_text("%s: "%node.firstChild.data)

    def do_parameterdefinition(self, node):
        self.generic_parse(node, pad=1)

    def do_detaileddescription(self, node):
        self.generic_parse(node, pad=1)

    def do_briefdescription(self, node):
        self.generic_parse(node, pad=1)

    def do_memberdef(self, node):
        prot = node.attributes['prot'].value
        id = node.attributes['id'].value
        kind = node.attributes['kind'].value
        tmp = node.parentNode.parentNode.parentNode
        compdef = tmp.getElementsByTagName('compounddef')[0]
        cdef_kind = compdef.attributes['kind'].value
        
        if prot == 'public':
            first = self.get_specific_nodes(node, ('definition', 'name'))
            name = first['name'].firstChild.data
            if name[:8] == 'operator': # Don't handle operators yet.
                return

            if self.include_function_definition:
                defn = first['definition'].firstChild.data
            else:
                defn = "" # Scip definition. Use the one produced with %feature("autodoc",1) instead
            self.add_text('\n')
            self.add_text('%feature("docstring") ')
            
            anc = node.parentNode.parentNode
            if cdef_kind in ('file', 'namespace'):
                ns_node = anc.getElementsByTagName('innernamespace')
                if not ns_node and cdef_kind == 'namespace':
                    ns_node = anc.getElementsByTagName('compoundname')
                if ns_node:
                    ns = ns_node[0].firstChild.data
                    self.add_text(' %s::%s "\n%s'%(ns, name, defn))
                else:
                    self.add_text(' %s "\n%s'%(name, defn))
            elif cdef_kind in ('class', 'struct'):
                # Get the full function name.
                anc_node = anc.getElementsByTagName('compoundname')
                cname = anc_node[0].firstChild.data
                self.add_text(' %s::%s "\n%s'%(cname, name, defn))

            for n in node.childNodes:
                if n not in first.values():
                    self.parse(n)
            self.add_text(['";', '\n'])
        
    def do_definition(self, node):
        data = node.firstChild.data
        self.add_text('%s "\n%s'%(data, data))

    def do_sectiondef(self, node):
        kind = node.attributes['kind'].value
        if kind in ('public-func', 'func'):
            self.generic_parse(node)

    def do_simplesect(self, node):
        kind = node.attributes['kind'].value
        if kind in ('date', 'rcs', 'version'):
            pass
        elif kind == 'warning':
            self.add_text(['\n', 'WARNING: '])
            self.generic_parse(node)
        elif kind == 'see':
            self.add_text('\n')
            self.add_text('See: ')
            self.generic_parse(node)
        else:
            self.generic_parse(node)

    def do_argsstring(self, node):
        self.generic_parse(node, pad=1)

    def do_member(self, node):
        kind = node.attributes['kind'].value
        refid = node.attributes['refid'].value
        if kind == 'function' and refid[:9] == 'namespace':
            self.generic_parse(node)

    def do_doxygenindex(self, node):
        self.multi = 1
        comps = node.getElementsByTagName('compound')
        for c in comps:
            refid = c.attributes['refid'].value
            fname = refid + '.xml'
            if not os.path.exists(fname):
                fname = os.path.join(self.my_dir,  fname)
            print "parsing file: %s"%fname
            p = Doxy2SWIG(fname)
            p.generate()
            self.pieces.extend(self.clean_pieces(p.pieces))

    def write(self, fname):
        o = my_open_write(fname)
        if self.multi:
            o.write("".join(self.pieces))
        else:
            o.write("".join(self.clean_pieces(self.pieces)))
        o.close()

    def clean_pieces(self, pieces):
        """Cleans the list of strings given as `pieces`.  It replaces
        multiple newlines by a maximum of 2 and returns a new list.
        It also wraps the paragraphs nicely.
        
        """
        ret = []
        count = 0
        for i in pieces:
            if i == '\n':
                count = count + 1
            else:
                if i == '";':
                    if count:
                        ret.append('\n')
                elif count > 2:
                    ret.append('\n\n')
                elif count:
                    ret.append('\n'*count)
                count = 0
                ret.append(i)

        _data = "".join(ret)
        ret = []
        for i in _data.split('\n\n'):
            if i == 'Parameters:':
                ret.extend(['Parameters:\n-----------', '\n\n'])
            elif i.find('// File:') > -1: # leave comments alone.
                ret.extend([i, '\n'])
            else:
                _tmp = textwrap.fill(i.strip())
                _tmp = self.lead_spc.sub(r'\1"\2', _tmp)
                ret.extend([_tmp, '\n\n'])
        return ret

class DocstringGenerator:
    def __init__(self, directory = os.curdir, header_files = "Use files in working directory", swig_directory = "swig",
                 docstring_file_base = "Use working directory", doxygen_file = "doxyfile", xml_directory  = "xml"):
        """
        A class to generate docstrings from c/c++ header files, using doxygen.

        Provide a directory that is scanned for header files or provide the header files directly
        as a list of strings.
        """
        # Set the name of the working directory
        if not os.path.isabs(directory):
            if directory == os.curdir:
                directory = os.path.abspath(os.curdir)
            else:
                directory = os.path.join(os.path.abspath(os.curdir),directory)
        self._directory = directory

        # Set the name of the header files, if not provided scan the working directory
        if header_files == "Use files in working directory":
            self._header_files = [ f for f in os.listdir(self._directory) if f.rfind(".h") != -1]
        else:
            self._header_files = header_files
            
        # Set the name of the directory where the generated interface file(s) are placed
        if os.path.isabs(swig_directory):
            self._swig_directory = swig_directory
        else:
            self._swig_directory = os.path.join(directory,swig_directory)

        # Set the name of the temporary doxyfile
        self._doxygen_file = os.path.join(directory, doxygen_file)

        # Set the name of the xml directory
        if os.path.isabs(xml_directory):
            self._xml_directory = xml_directory
        else:
            self._xml_directory = os.path.join(directory,xml_directory)

        # Set the name of the generated docstring file that should be %included in a swig
        # interface file before any %include "code" statements
        # The files name is: docstring_file_base + _docstrings.i
        if docstring_file_base == "Use working directory":
            # If the not provided then use current directory
            docstring_file_base = self._directory.split(os.path.sep)[-1]
        self._docstring_file = os.path.join(self._swig_directory,docstring_file_base + "_docstrings.i")
            
    def generate_doxygen_documentation(self):
        """
        Generate xml documentation with doxygen.
        Doxygen needs to be installed.
        """ 
        from subprocess import Popen
        header_files = " ".join(self._header_files)
        dfile = open(self._doxygen_file,'w')
        dfile.write("""INPUT  = %s
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
GENERATE_XML           = YES
XML_OUTPUT             = %s""" % (header_files,self._xml_directory))
        dfile.close()
        generate_doxygen_code = Popen(['doxygen',dfile.name])
        generate_doxygen_code.wait()
    
    def generate_interface_files_from_classes(self):
        """Only include class files when generating interface files"""
        from  doxy2swig import Doxy2SWIG
        class_files = [f for f in os.listdir(self._xml_directory) if f.find("class") != -1]
        dfile = open(self._docstring_file,"w")
        dfile.write("// An autogenerated docstringfile\n\n")
        dfile.close()
        for class_file in class_files:
            p = Doxy2SWIG(os.path.join(self._xml_directory,class_file))
            p.generate()
            dfile = open(self._docstring_file,"a")
            p.write(dfile)

    def generate_interface_file_from_index(self):
        """Take index.xml and generates one interface file."""
        from  doxy2swig import Doxy2SWIG
        p = Doxy2SWIG(os.path.join(self._xml_directory,"index.xml"))
        p.generate()
        dfile = open(self._docstring_file,"w")
        dfile.write("// An autogenerated docstringfile\n\n")
        #p.write(self._docstring_file)
        p.write(dfile)
    
    def clean(self):
        """Remove temporary xml directory and doxyfile."""
        for f in os.listdir(self._xml_directory):
            os.remove(os.path.join(self._xml_directory,f))
        os.removedirs(self._xml_directory)
        os.remove(self._doxygen_file)
        if 'html' in os.listdir(self._directory):
            html_dir = os.path.join(self._directory,"html")
            for f in os.listdir(html_dir):
                os.remove(os.path.join(html_dir,f))
            os.removedirs(html_dir)
            
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        directory = sys.argv[1]
        g =  DocstringGenerator(directory)
    else:
        g = DocstringGenerator()
    g.generate_doxygen_documentation()
    if "index.xml" in os.listdir(g._xml_directory):
        g.generate_interface_file_from_index()
    else:
        g.generate_interface_files_classes()
    g.clean()
