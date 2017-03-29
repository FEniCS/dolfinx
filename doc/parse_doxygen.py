#!/usr/bin/env python
#
# Read doxygen xml files into an object tree that can be
# written to either Sphinx ReStructuredText format or SWIG doctrings
#
# Written by Tormod Landet, 2017
#
from __future__ import print_function
import sys, os


try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET


MOCK_REPLACEMENTS = {'operator[]': '__getitem__',
                     'operator()': '__call__',
                     'in': '_in',
                     'self': '_self',
                     'string': 'str',
                     'false': 'False',
                     'true': 'True',
                     'null': 'None'}


class Namespace(object):
    """
    A C++ namespace
    """
    def __init__(self, name):
        self.name = name # We use '' for the global namespace
        self.parent_namespace = None
        self.subspaces = {}
        self.members = {}
    
    def add(self, item):
        """
        Add a member to this namespace
        """
        if isinstance(item, Namespace):
            self.subspaces[item.name] = item
            item.parent_namespace = self
        else:
            self.members[item.name] = item
            item.namespace = self
    
    def lookup(self, name):
        """
        Find a named member. This is currently only used to find classes
        in namespaces. We use this to find the superclasses of each class
        since we only know their names and we need the list their
        superclasses as well to show only direct superclasses when we
        generate documentation for each class  
        """
        item = self.members.get(name, None)
        if item: return item
        
        nn = name.split('::')[0]
        
        if nn == self.name:
            return None
        
        if nn in self.subspaces:
            return self.subspaces[nn].lookup(item)
        
        if self.parent_namespace is not None:
            return self.parent_namespace.lookup(name)


class Parameter(object):
    """
    A parameter to a C++ function
    """
    def __init__(self, name, param_type, value=None, description=''):
        self.name = name
        self.type = param_type
        self.value = None
        self.description = description
    
    @staticmethod
    def from_param(param_elem):
        """
        Read a <param> element and get name and type
        """
        tp = get_single_element(param_elem, 'type') 
        param_type = description_to_string(tp, skipelems=('ref',))
        
        n1 = get_single_element(param_elem, 'declname', True)
        n2 = get_single_element(param_elem, 'defname', True)
        if n1 is None and n2 is not None:
            name = n2.text
        elif n1 is None:
            name = ''
        else:
            name = n1.text
        
        defval = None
        dv = get_single_element(param_elem, 'defval', True)
        if dv is not None:
            defval = dv.text
        
        description = ''
        doc1 = get_single_element(param_elem, 'briefdescription', True)
        if doc1 is not None:
            description = description_to_string(doc1)
        
        return Parameter(name, param_type, defval, description)


class NamespaceMember(object):
    """
    A class, function, enum, struct, or typedef
    """
    
    def __init__(self, name, kind):
        # Used for every member
        self.name = name
        self.kind = kind  # enum, struct, function or class
        self.namespace = None
        self.docstring = []
        self.short_name = None
        self.protection = None
        self.hpp_file_name = None
        self.xml_file_name = None
        
        # Used for functions and variables
        self.type = None
        self.type_description = ''

        # Used for functions
        self.parameters = []
        
        # Used for classes
        self.superclasses = []
        self.members = {}
        
        # Used for enums
        self.enum_values = []
    
    def add(self, item):
        """
        Add a member to a class
        """
        assert self.kind == 'class'
        self.members[item.name] = item
        item.namespace = self
    
    @staticmethod
    def from_memberdef(mdef, name, kind, xml_file_name, namespace_obj):
        """
        Read a memberdef element from a nameclass file (members of
        the namespace that are not classes) or from a compoundef
        (members of the class)
        """
        # Make sure we have the required "namespace::" prefix
        required_prefix = '%s::' % namespace_obj.name
        if kind != 'friend' and not name.startswith(required_prefix):
            name = required_prefix + name
        short_name = name[len(required_prefix):]
        
        if kind == 'function':
            argsstring = get_single_element(mdef, 'argsstring').text
            name += argsstring
        
        # Define attributes common to all namespace members
        item = NamespaceMember(name, kind)
        item.short_name = short_name
        item.protection = mdef.attrib['prot']
        item.xml_file_name = xml_file_name
        item.hpp_file_name = get_single_element(mdef, 'location').attrib['file']
        item._add_doc(mdef)
        
        # Get parameters (for functions)
        for param in mdef.findall('param'):
            item.parameters.append(Parameter.from_param(param))
            
        # Get type (return type for functions)
        mtype = get_single_element(mdef, 'type', True)
        if mtype is not None:
            item.type = description_to_string(mtype, skipelems=('ref',))
            
        # Get parameter descriptions
        dd = get_single_element(mdef, 'detaileddescription')
        for pi in findall_recursive(dd, 'parameteritem'):
            pnl = get_single_element(pi, 'parameternamelist')
            pd = get_single_element(pi, 'parameterdescription')
            param_desc = description_to_string(pd)
            
            pns = pnl.findall('parametername')
            for pn in pns:
                pname = pn.text
                pdesc = param_desc
                if 'direction' in pn.attrib:
                    pdesc += ' [direction=%s]' % pn.attrib['direction']
                maching_params = [p for p in item.parameters if p.name == pname]
                assert len(maching_params) == 1
                maching_params[0].description += ' ' + pdesc
        
        # Get return type description
        for ss in findall_recursive(dd, 'simplesect'):
            memory = {'skip_simplesect': False}
            if ss.get('kind', '') == 'return':
                item.type_description = description_to_string(ss, memory=memory)
        
        # Get enum values
        for ev in mdef.findall('enumvalue'):
            ename = get_single_element(ev, 'name').text
            evalue = ''
            init = get_single_element(ev, 'initializer', True)
            if init is not None:
                evalue = init.text
            item.enum_values.append((ename, evalue))
        
        return item
    
    @staticmethod
    def from_compounddef(cdef, name, kind, xml_file_name):
        """
        Read a compounddef element from a class definition file
        """
        item = NamespaceMember(name, kind)
        item.hpp_file_name = get_single_element(cdef, 'location').attrib['file']
        item.xml_file_name = xml_file_name
        item.short_name = name.split('::')[-1]
        item._add_doc(cdef)
        
        # Get superclasses with public inheritance
        igs = cdef.findall('collaborationgraph')
        if len(igs) == 1:
            for node in igs[0]:
                public = False
                for cn in node.findall('childnode'):
                    if cn.attrib.get('relation', '') == 'public-inheritance':
                        public = True
                if public:
                    label = get_single_element(node, 'label').text
                    if label != item.name:
                        item.superclasses.append(label)
        else:
            assert len(igs) == 0
            
        # Read members
        for s in cdef.findall('sectiondef'):
            members = s.findall('memberdef')
            for m in members:
                mname = get_single_element(m, 'name').text
                mkind = m.attrib['kind']
                mitem = NamespaceMember.from_memberdef(m, mname, mkind, xml_file_name, item)
                item.add(mitem)
        
        return item
    
    def _add_doc(self, elem):
        """
        Add docstring for the given element
        """
        bd = get_single_element(elem, 'briefdescription')
        dd = get_single_element(elem, 'detaileddescription')
        
        description_to_rst(bd, self.docstring)
        if self.docstring and self.docstring[-1].strip():
            self.docstring.append('')
        description_to_rst(dd, self.docstring)
    
    def get_superclasses(self):
        to_remove = set()
        for sn in self.superclasses:
            obj = self.namespace.lookup(sn)
            if obj:
                for sn2 in obj.superclasses:
                    if sn2 in self.superclasses:
                        to_remove.add(sn2)
        return [sn for sn in self.superclasses if sn not in to_remove]
    
    def _to_rst_string(self, indent, for_swig=False, for_mock=False):
        """
        Create a list of lines on Sphinx (ReStructuredText) format
        The header is included for Sphinx, but not for SWIG docstrings
        """
        ret = []
        if for_mock:
            for_swig = True
        
        if not for_swig:
            ret.append('')
            simple_kinds = {'typedef': 'type', 'enum': 'enum'}
            kinds_with_types = {'function': 'function', 'variable': 'var'}
            if self.kind in ('class', 'struct'):
                superclasses = self.get_superclasses()
                supers = ''
                if superclasses:
                    supers = ' : public ' + ', '.join(superclasses)
                ret.append(indent + '.. cpp:class:: %s%s' % (self.name, supers))
            elif self.kind in kinds_with_types:
                rst_kind = kinds_with_types[self.kind]
                ret.append(indent + '.. cpp:%s:: %s %s' % (rst_kind, self.type, self.name))
            elif self.kind in simple_kinds:
                rst_kind = simple_kinds[self.kind]
                ret.append(indent + '.. cpp:%s:: %s' % (rst_kind, self.name))
            else:
                raise NotImplementedError('Kind %s not implemented' % self.kind)
            
            # Docstring and parameters must be further indented
            indent += '   '
            ret.append(indent)
        
        # All: add docstring
        doclines = self.docstring
        if doclines and not doclines[0].strip():
            doclines = doclines[1:]
        ret.extend(indent + line for line in doclines)
        
        # Classes: separate friends from other members
        friends, members = [], []
        for _, member in sorted(self.members.items()):
            if member.kind == 'friend':
                friends.append(member)
            else:
                members.append(member)
                
        # Classes: add friends
        if friends:
            if ret and ret[-1].strip():
                ret.append(indent)
            ret.append(indent + 'Friends: %s.' % ', '.join(
                ':cpp:any:`%s`' % friend.name for friend in friends))
        
        # Functions: add space before parameters and return types
        if self.parameters or (self.type and for_swig) or self.type_description:
            if ret and ret[-1].strip():
                ret.append(indent)
        
        # Functions: add parameter definitions
        for param in self.parameters:
            if param.name:
                ptype = param.type.replace(':', '\\:')
                pname = param.name.replace(':', '\\:')
                if for_swig:
                    ret.append(indent + ':param %s %s: %s' % (ptype, pname, param.description))                    
                else:
                    # Parameter type is redundant info in the Sphinx produced html
                    ret.append(indent + ':param %s: %s' % (pname, param.description))
        
        # Functions: add return type (redundant info in the Sphinx produced html)
        if self.type and for_swig:
            ret.append(indent + ':rtype: %s' % self.type)
        if self.type_description:
            ret.append(indent + ':returns: %s' % self.type_description)
            
        # Enums: add enumerators
        for ename, evalue in self.enum_values:
            if ret and ret[-1].strip():
                ret.append(indent)
            ret.append(indent + '.. cpp:enumerator:: %s::%s %s' % (self.name, ename, evalue))
            ret.append(indent)
        
        # Remove doubled up blank lines
        if ret:
            ret2 = [ret[0]]
            for line in ret[1:]:
                if line.strip() or ret2[-1].strip():
                    ret2.append(line)
            ret = ret2
        
        # All: SWIG items are not nested, so we end this one here    
        if for_swig and not for_mock:
            escaped = [line.replace('\\', '\\\\').replace('"', '\\"') for line in ret]
            sname = self.name
            if sname.endswith('=0'):
                sname = sname[:-2]
            elif sname.endswith('=delete'):
                sname = sname[:-7]
            ret = ['%%feature("docstring") %s "' % sname,
                   '\n'.join(escaped).rstrip() + '\n";\n']
            indent = ''
        
        # Classes: add members of a class
        if not for_mock:
            for member in members:
                ret.append(member._to_rst_string(indent, for_swig))
        
        return '\n'.join(ret)
    
    def to_swig(self):
        """
        Output SWIG docstring definitions
        """
        swigstr = self._to_rst_string(indent='', for_swig=True)
        # Get rid of repeated newlines
        for _ in range(4):
            swigstr = swigstr.replace('\n\n\n', '\n\n')
        return swigstr
    
    def to_rst(self, indent=''):
        """
        Output Sphinx :cpp:*:: definitions
        """
        return self._to_rst_string(indent)
    
    def to_mock(self, modulename, indent='', _classname=None):
        """
        Output mock Python code
        """
        # Get swig docstring and get rid of any repeated newlines
        new_indent = indent + '    '
        swigdoc = self._to_rst_string(indent=new_indent, for_mock=True)
        for _ in range(4):
            swigdoc = swigdoc.replace('\n\n\n', '\n\n')
        
        ret = []
        if self.kind == 'class':
            superclasses = [sc.split('::')[-1].split('<')[0] for sc in self.get_superclasses()]
            superclasses = [MOCK_REPLACEMENTS.get(sc, sc) for sc in superclasses]
            superclasses = [sc for sc in superclasses if sc != self.short_name]
            ret.append(indent + 'class %s:\n' % (self.short_name))
            ret.append(new_indent + '"""\n')
            ret.append(swigdoc.rstrip())
            ret.append('\n')
            for sc in superclasses:
                ret.append(new_indent + 'Superclass: :py:obj:`%s`\n' % sc)
            ret.append(new_indent + '"""\n')
            for member in self.members.values():
                ret.append('\n')
                ret.append(member.to_mock(modulename, indent=new_indent, _classname=self.short_name))
        
        elif self.kind == 'function':
            fname = MOCK_REPLACEMENTS.get(self.short_name, self.short_name)
            if fname.startswith('operator'):
                return ''
            params = [MOCK_REPLACEMENTS.get(p.name, p.name) for p in self.parameters]
            if _classname:
                params = ['self'] + params
                if _classname == fname:
                    fname = '__init__'
                elif '~' + _classname == fname:
                    fname = '__del__'
            
            if self.parameters and self.parameters[-1].type == '...':
                params[-1] = '*args'
            args = ', '.join(params)
            
            ret.append(indent + 'def %s(%s):\n' % (fname, args))
            ret.append(new_indent + '"""\n')
            ret.append(swigdoc.rstrip())
            ret.append('\n')
            ret.append(new_indent + '"""\n')
            ret.append(new_indent + 'print(WARNING)\n')
        
        elif self.kind == 'friend':
            pass
        
        elif self.kind == 'enum':
            ret.append(indent + '# Enumeration %s\n' % self.short_name)
            for ename, evalue in self.enum_values:
                ret.append(indent + '#: Enum value %s::%s %s\n' % (self.name, ename, evalue))
                if not evalue.strip():
                    evalue = '= None'
                evalue = MOCK_REPLACEMENTS.get(evalue[1:].strip(), evalue[1:].strip())
                ret.append(indent + '%s.%s_%s = %s\n' % (modulename, self.short_name, ename, evalue))
        
        elif self.kind == 'variable':
            pass
            #for line in swigdoc.split('\n'):
            #    ret.append(indent + '#: %s\n' % line)
            #ret.append(indent + 'module.%s = None\n' % self.short_name)
        
        elif self.kind == 'typedef':
            pass
        
        else:
            print('WARNING: kind %s not supported by to_mock()' % self.kind)
        
        if not _classname and self.kind in ('class', 'function'):
            ret.append('%s.%s = %s\n' % (modulename, self.short_name, self.short_name))

        return ''.join(ret)
    
    def __str__(self):
        return self.to_rst()


def description_to_string(element, indent='', skipelems=(), memory=None):
    lines = []
    description_to_rst(element, lines, indent, skipelems, memory)
    return ' '.join(lines).strip()


NOT_IMPLEMENTED_ELEMENTS = set()
def description_to_rst(element, lines, indent='', skipelems=(), memory=None):
    """
    Create a valid ReStructuredText block for Sphinx from the given description
    element. Handles <para> etc markup inside and also is called for every
    sub-tag of the description tag (like <para>, <ref> etc)
    """
    if lines == []:
        lines.append('')
    if memory is None:
        memory = dict()
    
    tag = element.tag
    children = list(element)
    postfix = ''
    postfix_lines = []
    
    # Handle known tags that show up in description type element trees
    # Tag contents are handled beneath, if the if-branch does not return
    # Unknown tags are just output unchanged and a WARNING is shown (in main)
    if tag in ('briefdescription', 'detaileddescription', 'parameterdescription',
               'type', 'highlight'):
        pass
    
    elif element in memory:
        # This element is being re-read, only treat the contained elements
        pass
    
    elif tag == 'para':
        if lines[-1].strip():
            lines.append(indent)
        postfix_lines.append(indent)
    
    elif tag == 'codeline':
        memory = dict(memory); memory[element] = 1
        skipelems = set(skipelems); skipelems.add('ref')
        line = description_to_string(element, indent, skipelems, memory)
        if line.startswith('*'):
            line = line[1:]
        lines.append(indent + line)
        return
    
    elif tag == 'ndash':
        lines[-1] += '--'
    
    elif tag == 'mdash':
        lines[-1] += '---'
    
    elif tag == 'sp':
        lines[-1] += ' '
    
    elif tag == 'ref':
        if 'ref' in skipelems:
            lines[-1] += element.text
        else:
            lines[-1] += ':cpp:any:`%s` ' % element.text
        return
    
    elif tag == 'ulink':
        lines[-1] += '`%s <%s>`_ ' % (element.text, element.get('url'))
        return
    
    elif tag == 'emphasis':
        if children and children[0].tag != 'ref':
            lines[-1] += '**'
            postfix += '**'
    
    elif tag == 'computeroutput':
        if element.text is None and not list(element):
            return
        elif lines[-1].endswith(':math:'):
            lines[-1] += '`'
            postfix += '` '
        else:
            lines[-1] += '``'
            postfix += '`` '
    
    elif tag in ('verbatim', 'programlisting'):
        if element.text is None and not list(element):
            return
        if lines[-1].strip():
            lines.append(indent)
        lines.append(indent + '::')
        postfix_lines.append(indent)
        indent += '   '
        lines.append(indent)
        lines.append(indent)
    
    elif tag == 'table':
        lines.extend(xml_table_to_rst(element, indent))
        lines.append(indent)
        return # Do not process children
    
    elif tag == 'formula':
        contents = element.text.strip()
        if contents.startswith(r'\['):
            if lines[-1].strip():
                lines.append(indent)
            lines.append(indent + '.. math::')
            lines.append(indent + '   ')
            lines.append(indent + '   ' + contents[2:-2])
            lines.append(indent)
            lines.append(indent)
        else:
            lines[-1] += ' :math:`' + contents[1:-1] + '` '
        return
    
    elif tag == 'itemizedlist':
        memory = dict(memory)
        memory['list_item_prefix'] = '*  '
        if lines[-1].strip():
            lines.append(indent)
        postfix_lines.append(indent)
    
    elif tag == 'orderedlist':
        memory = dict(memory)
        memory['list_item_prefix'] = '#. '
        if lines[-1].strip():
            lines.append(indent)
        postfix_lines.append(indent)
    
    elif tag == 'listitem':
        memory = dict(memory); memory[element] = 1
        item = description_to_string(element, indent + '   ', skipelems, memory)
        lines.append(indent + memory['list_item_prefix'] + item)
        return
    
    elif tag == 'parameterlist':
        # We parse these separately in the parameter reading process
        return
    
    elif tag == 'simplesect' and element.get('kind', '') == 'return':
        # We parse these separately in the return-type reading process
        if memory.get('skip_simplesect', True):
            return
    
    else:
        NOT_IMPLEMENTED_ELEMENTS.add(tag)
        lines.append(ET.tostring(element)) 
    
    def add_text(text):
        if text is not None and text.strip():
            tl = text.split('\n')
            lines[-1] += tl[0]
            lines.extend([indent + line for line in tl[1:]])
            if text.endswith('\n'):
                lines.append(indent)
    
    add_text(element.text)
    
    for child in children:
        description_to_rst(child, lines, indent, skipelems, memory)
        add_text(child.tail)
    
    if postfix:
        lines[-1] += postfix
    
    if postfix_lines:
        lines.extend(postfix_lines)


def xml_table_to_rst(table, indent):
    """
    Read an XML table element and produce a ReStructuredText table
    
    =====  ====
    Col A  B
    =====  ====
    hi     1.00
    there  3.14
    =====  ====
    """
    rows = int(table.get('rows'))
    cols = int(table.get('cols'))
    data = [[''] * cols for row in range(rows)]
    widths = [0] * cols
    for i, row in enumerate(table):
        for j, entry in enumerate(row):
            memory = {entry: 1}
            text = description_to_string(entry, indent='', 
                                         skipelems=('para',),
                                         memory=memory)
            widths[j] = max(widths[j], len(text))
            data[i][j] = text
    
    ret = [indent] * (rows + 3)
    for w in widths:
        ret[0] += '=' * w + '  '
        ret[2] += '=' * w + '  '
        ret[-1] += '=' * w + '  '
    
    for i, row in enumerate(data):
        I = 1 if i == 0 else i + 2
        for j, text in enumerate(row):
            ret[I] += text + ' ' * (widths[j] + 2 - len(text))
    
    return ret


def get_single_element(parent, name, allow_none=False):
    """
    Helper to get one unique child element
    """
    elems = parent.findall(name)
    N = len(elems)
    if N != 1:
        if allow_none and N == 0:
            return None 
        raise ValueError('Expected one element %r below %r, got %r' % 
                         (name, parent, len(elems)))
    return elems[0]


def findall_recursive(parent, name):
    for item in parent.findall(name):
        yield item
    for child in parent:
        for item in findall_recursive(child, name):
            yield item


def read_doxygen_xml_files(xml_directory, namespace_names, verbose=True):
    """
    Read doxygen XML files from the given directory. Restrict the returned
    namespaces to the ones listed in the namespaces input iterable
    
    Remember: we are built for speed, not ultimate flexibility, hence the
    restrictions to avoid parsing more than we are interested in actually
    outputing in the end 
    """
    if verbose: print('Parsing doxygen XML files in %s' % xml_directory)
    
    root_namespace = Namespace('')
    for nn in namespace_names:
        root_namespace.add(Namespace(nn))
    
    # Loop through xml files of compounds and get class definitions
    xml_files = os.listdir(xml_directory)
    for xml_file_name in xml_files:
        if not xml_file_name.startswith('class'):
            continue
        
        path = os.path.join(xml_directory, xml_file_name)
        try:
            root = ET.parse(path).getroot()
        except Exception as e:
            print('ERROR parsing doxygen xml document', path)
            print(e)
            continue
        
        compounds = root.findall('compounddef')
        for c in compounds:
            kind = c.attrib['kind']
            names = c.findall('compoundname')
            
            assert len(names) == 1
            name = names[0].text
            nn = name.split('::')[0]
            namespace = root_namespace.subspaces.get(nn, None)
            if not namespace:
                continue
            
            item = NamespaceMember.from_compounddef(c, name, kind, xml_file_name)
            namespace.add(item)
            
            if verbose: print(end='.')
    if verbose: print('DONE\nParsing namespace files:')
    
    # Loop through other elements in the namespaces
    for namespace in root_namespace.subspaces.values():
        file_name = 'namespace%s.xml' % namespace.name
        path = os.path.join(xml_directory, file_name)
        try:
            root = ET.parse(path).getroot()
        except Exception as e:
            print('ERROR parsing doxygen xml document', path)
            print(e)
            continue
        
        compound = get_single_element(root, 'compounddef')
        sections = compound.findall('sectiondef')
        for s in sections:
            members = s.findall('memberdef')
            for m in members:
                name = get_single_element(m, 'name').text
                kind = m.attrib['kind']
                item = NamespaceMember.from_memberdef(m, name, kind, xml_file_name, namespace)
                namespace.add(item)
        if verbose: print(' - ', namespace.name)  
    if verbose: print('Done parsing files')
    
    return root_namespace.subspaces


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Call me like "script path_to/xml/dir namespace"')
        print('An exampe:\n\tpython parse_doxygen.py doxygen/xml dolfin')
        print('ERROR: I need two arguments!')
        exit(1)
    
    xml_directory = sys.argv[1]
    namespace = sys.argv[2]
    
    # Parse the XML files
    namespaces = read_doxygen_xml_files(xml_directory, [namespace])
    
    # Get sorted list of members
    members = list(namespaces[namespace].members.values())
    members.sort(key=lambda m: m.name)
    
    # Make Sphinx documentation
    with open('api.rst', 'wt') as out:
        for member in members:
            out.write(member.to_rst())
            out.write('\n')
        out.write('\n')
    
    # Make SWIG interface file
    with open('docstrings.i', 'wt') as out:
        out.write('// SWIG docstrings generated by doxygen and parse_doxygen.py\n\n')
        for member in members:
            out.write(member.to_swig())
            out.write('\n')
        out.write('\n')
    
    for tag in NOT_IMPLEMENTED_ELEMENTS:
        print('WARNING: doxygen XML tag %s is not supported by the parser' % tag)
