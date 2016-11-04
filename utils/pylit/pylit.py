#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# pylit.py
# ********
# Literate programming with reStructuredText
# ++++++++++++++++++++++++++++++++++++++++++
#
# :Date:      $Date$
# :Revision:  $Revision$
# :URL:       $URL$
# :Copyright: © 2005, 2007 Günter Milde.
#             Released without warranty under the terms of the
#             GNU General Public License (v. 2 or later)
#
# ::

from __future__ import print_function

"""pylit: bidirectional text <-> code converter

Convert between a *text document* with embedded code
and *source code* with embedded documentation.
"""

# .. contents::
#
# Frontmatter
# ===========
#
# Changelog
# ---------
#
# .. class:: borderless
#
# ======  ==========  ===========================================================
# 0.1     2005-06-29  Initial version.
# 0.1.1   2005-06-30  First literate version.
# 0.1.2   2005-07-01  Object orientated script using generators.
# 0.1.3   2005-07-10  Two state machine (later added 'header' state).
# 0.2b    2006-12-04  Start of work on version 0.2 (code restructuring).
# 0.2     2007-01-23  Published at http://pylit.berlios.de.
# 0.2.1   2007-01-25  Outsourced non-core documentation to the PyLit pages.
# 0.2.2   2007-01-26  New behaviour of `diff` function.
# 0.2.3   2007-01-29  New `header` methods after suggestion by Riccardo Murri.
# 0.2.4   2007-01-31  Raise Error if code indent is too small.
# 0.2.5   2007-02-05  New command line option --comment-string.
# 0.2.6   2007-02-09  Add section with open questions,
#                     Code2Text: let only blank lines (no comment str)
#                     separate text and code,
#                     fix `Code2Text.header`.
# 0.2.7   2007-02-19  Simplify `Code2Text.header`,
#                     new `iter_strip` method replacing a lot of ``if``-s.
# 0.2.8   2007-02-22  Set `mtime` of outfile to the one of infile.
# 0.3     2007-02-27  New `Code2Text` converter after an idea by Riccardo Murri,
#                     explicit `option_defaults` dict for easier customisation.
# 0.3.1   2007-03-02  Expand hard-tabs to prevent errors in indentation,
#                     `Text2Code` now also works on blocks,
#                     removed dependency on SimpleStates module.
# 0.3.2   2007-03-06  Bug fix: do not set `language` in `option_defaults`
#                     renamed `code_languages` to `languages`.
# 0.3.3   2007-03-16  New language css,
#                     option_defaults -> defaults = optparse.Values(),
#                     simpler PylitOptions: don't store parsed values,
#                     don't parse at initialisation,
#                     OptionValues: return `None` for non-existing attributes,
#                     removed -infile and -outfile, use positional arguments.
# 0.3.4   2007-03-19  Documentation update,
#                     separate `execute` function.
#         2007-03-21  Code cleanup in `Text2Code.__iter__`.
# 0.3.5   2007-03-23  Removed "css" from known languages after learning that
#                     there is no C++ style "// " comment string in CSS2.
# 0.3.6   2007-04-24  Documentation update.
# 0.4     2007-05-18  Implement Converter.__iter__ as stack of iterator
#                     generators. Iterating over a converter instance now
#                     yields lines instead of blocks.
#                     Provide "hooks" for pre- and postprocessing filters.
#                     Rename states to reduce confusion with formats:
#                     "text" -> "documentation", "code" -> "code_block".
# 0.4.1   2007-05-22  Converter.__iter__: cleanup and reorganisation,
#                     rename parent class Converter -> TextCodeConverter.
# 0.4.2   2007-05-23  Merged Text2Code.converter and Code2Text.converter into
#                     TextCodeConverter.converter.
# 0.4.3   2007-05-30  Replaced use of defaults.code_extensions with
#                     values.languages.keys().
#                     Removed spurious `print` statement in code_block_handler.
#                     Added basic support for 'c' and 'css' languages
#                     with `dumb_c_preprocessor`_ and `dumb_c_postprocessor`_.
# 0.5     2007-06-06  Moved `collect_blocks`_ out of `TextCodeConverter`_,
#                     bug fix: collect all trailing blank lines into a block.
#                     Expand tabs with `expandtabs_filter`_.
# 0.6     2007-06-20  Configurable code-block marker (default ``::``)
# 0.6.1   2007-06-28  Bug fix: reset self.code_block_marker_missing.
# 0.7     2007-12-12  prepending an empty string to sys.path in run_doctest()
#                     to allow imports from the current working dir.
# 0.7.1   2008-01-07  If outfile does not exist, do a round-trip conversion
#                     and report differences (as with outfile=='-').
# 0.7.2   2008-01-28  Do not add missing code-block separators with
#                     `doctest_run` on the code source. Keeps lines consistent.
# 0.7.3   2008-04-07  Use value of code_block_marker for insertion of missing
#                     transition marker in Code2Text.code_block_handler
#                     Add "shell" to defaults.languages
# 0.7.4   2008-06-23  Add "latex" to defaults.languages
# 0.7.5   2009-05-14  Bugfix: ignore blank lines in test for end of code block
# 0.7.6   2009-12-15  language-dependent code-block markers (after a
#                     `feature request and patch by jrioux`_),
#                     use DefaultDict for language-dependent defaults,
#                     new defaults setting `add_missing_marker`_.
# 0.7.7   2010-06-23  New command line option --codeindent.
# 0.7.8   2011-03-30  bugfix: do not overwrite custom `add_missing_marker` value,
#                     allow directive options following the 'code' directive.
# 0.7.9   2011-04-05  Decode doctest string if 'magic comment' gives encoding.
# ======  ==========  ===========================================================
#
# ::

_version = "0.7.9"

__docformat__ = 'restructuredtext'


# Introduction
# ------------
#
# PyLit is a bidirectional converter between two formats of a computer
# program source:
#
# * a (reStructured) text document with program code embedded in
#   *code blocks*, and
# * a compilable (or executable) code source with *documentation*
#   embedded in comment blocks
#
#
# Requirements
# ------------
#
# ::

import os, sys
import re, optparse


# DefaultDict
# ~~~~~~~~~~~
# As `collections.defaultdict` is only introduced in Python 2.5, we
# define a simplified version of the dictionary with default from
# http://code.activestate.com/recipes/389639/
# ::

class DefaultDict(dict):
    """Minimalistic Dictionary with default value."""
    def __init__(self, default=None, *args, **kwargs):
        self.update(dict(*args, **kwargs))
        self.default = default

    def __getitem__(self, key):
        return self.get(key, self.default)


# Defaults
# ========
#
# The `defaults` object provides a central repository for default
# values and their customisation. ::

defaults = optparse.Values()

# It is used for
#
# * the initialisation of data arguments in TextCodeConverter_ and
#   PylitOptions_
#
# * completion of command line options in `PylitOptions.complete_values`_.
#
# This allows the easy creation of back-ends that customise the
# defaults and then call `main`_ e.g.:
#
# >>> import pylit
# >>> pylit.defaults.comment_string = "## "
# >>> pylit.defaults.codeindent = 4
# >>> pylit.main()
#
# The following default values are defined in pylit.py:
#
# languages
# ---------
#
# Mapping of code file extensions to code language::

defaults.languages  = DefaultDict("python", # fallback language
                                  {".c":   "c",
                                   ".cc":  "c++",
                                   ".cpp": "c++",
                                   ".css": "css",
                                   ".py":  "python",
                                   ".sh":  "shell",
                                   ".sl":  "slang",
                                   ".sty": "latex",
                                   ".tex": "latex",
                                   ".ufl": "python"
                                  })

# Will be overridden by the ``--language`` command line option.
#
# The first argument is the fallback language, used if there is no
# matching extension (e.g. if pylit is used as filter) and no
# ``--language`` is specified. It can be changed programmatically by
# assignment to the ``.default`` attribute, e.g.
#
# >>> defaults.languages.default='c++'
#
#
# .. _text_extension:
#
# text_extensions
# ---------------
#
# List of known extensions of (reStructured) text files. The first
# extension in this list is used by the `_get_outfile_name`_ method to
# generate a text output filename::

defaults.text_extensions = [".txt", ".rst"]


# comment_strings
# ---------------
#
# Comment strings for known languages. Used in Code2Text_ to recognise
# text blocks and in Text2Code_ to format text blocks as comments.
# Defaults to ``'# '``.
#
# **Comment strings include trailing whitespace.** ::

defaults.comment_strings = DefaultDict('# ',
                                       {"css":    '// ',
                                        "c":      '// ',
                                        "c++":    '// ',
                                        "latex":  '% ',
                                        "python": '# ',
                                        "shell":  '# ',
                                        "slang":  '% '
                                       })


# header_string
# -------------
#
# Marker string for a header code block in the text source. No trailing
# whitespace needed as indented code follows.
# Must be a valid rst directive that accepts code on the same line, e.g.
# ``'..admonition::'``.
#
# Default is a comment marker::

defaults.header_string = '..'


# .. _code_block_marker:
#
# code_block_markers
# ------------------
#
# Markup at the end of a documentation block.
# Default is Docutils' marker for a `literal block`_::

defaults.code_block_markers = DefaultDict('::')
defaults.code_block_markers["c++"] = u".. code-block:: cpp"
#defaults.code_block_markers['python'] = '.. code-block:: python'

# The `code_block_marker` string is `inserted into a regular expression`_.
# Language-specific markers can be defined programmatically, e.g. in a
# wrapper script.
#
# In a document where code examples are only one of several uses of
# literal blocks, it is more appropriate to single out the source code
# ,e.g. with the double colon at a separate line ("expanded form")
#
#   ``defaults.code_block_marker.default = ':: *'``
#
# or a dedicated ``.. code-block::`` directive [#]_
#
#   ``defaults.code_block_marker['c++'] = '.. code-block:: *c++'``
#
# The latter form also allows code in different languages kept together
# in one literate source file.
#
# .. [#] The ``.. code-block::`` directive is not (yet) supported by
#    standard Docutils.  It is provided by several add-ons, including
#    the `code-block directive`_ project in the Docutils Sandbox and
#    Sphinx_.
#
#
# strip
# -----
#
# Export to the output format stripping documentation or code blocks::

defaults.strip = False

# strip_marker
# ------------
#
# Strip literal marker from the end of documentation blocks when
# converting  to code format. Makes the code more concise but looses the
# synchronisation of line numbers in text and code formats. Can also be used
# (together with the auto-completion of the code-text conversion) to change
# the `code_block_marker`::

defaults.strip_marker = False

# add_missing_marker
# ------------------
#
# When converting from code format to text format, add a `code_block_marker`
# at the end of documentation blocks if it is missing::

defaults.add_missing_marker = True

# Keep this at ``True``, if you want to re-convert to code format later!
#
#
# .. _defaults.preprocessors:
#
# preprocessors
# -------------
#
# Preprocess the data with language-specific filters_
# Set below in Filters_::

defaults.preprocessors = {}

# .. _defaults.postprocessors:
#
# postprocessors
# --------------
#
# Postprocess the data with language-specific filters_::

defaults.postprocessors = {}

# .. _defaults.codeindent:
#
# codeindent
# ----------
#
# Number of spaces to indent code blocks in `Code2Text.code_block_handler`_::

defaults.codeindent =  2

# In `Text2Code.code_block_handler`_, the codeindent is determined by the
# first recognised code line (header or first indented literal block
# of the text source).
#
# overwrite
# ---------
#
# What to do if the outfile already exists? (ignored if `outfile` == '-')::

defaults.overwrite = 'yes'

# Recognised values:
#
#  :'yes':    overwrite eventually existing `outfile`,
#  :'update': fail if the `outfile` is newer than `infile`,
#  :'no':     fail if `outfile` exists.
#
#
# Extensions
# ==========
#
# Try to import optional extensions::

try:
    import pylit_elisp
except ImportError:
    pass


# Converter Classes
# =================
#
# The converter classes implement a simple state machine to separate and
# transform documentation and code blocks. For this task, only a very limited
# parsing is needed. PyLit's parser assumes:
#
# * `indented literal blocks`_ in a text source are code blocks.
#
# * comment blocks in a code source where every line starts with a matching
#   comment string are documentation blocks.
#
# TextCodeConverter
# -----------------
# ::

class TextCodeConverter(object):
    """Parent class for the converters `Text2Code` and `Code2Text`.
    """

# The parent class defines data attributes and functions used in both
# `Text2Code`_ converting a text source to executable code source, and
# `Code2Text`_ converting commented code to a text source.
#
# Data attributes
# ~~~~~~~~~~~~~~~
#
# Class default values are fetched from the `defaults`_ object and can be
# overridden by matching keyword arguments during class instantiation. This
# also works with keyword arguments to `get_converter`_ and `main`_, as these
# functions pass on unused keyword args to the instantiation of a converter
# class. ::

    language = defaults.languages.default
    comment_strings = defaults.comment_strings
    comment_string = "" # set in __init__ (if empty)
    codeindent =  defaults.codeindent
    header_string = defaults.header_string
    code_block_markers = defaults.code_block_markers
    code_block_marker = "" # set in __init__ (if empty)
    strip = defaults.strip
    strip_marker = defaults.strip_marker
    add_missing_marker = defaults.add_missing_marker
    directive_option_regexp = re.compile(r' +:(\w|[-._+:])+:( |$)')
    state = "" # type of current block, see `TextCodeConverter.convert`_

# Interface methods
# ~~~~~~~~~~~~~~~~~
#
# .. _TextCodeConverter.__init__:
#
# __init__
# """"""""
#
# Initialising sets the `data` attribute, an iterable object yielding lines of
# the source to convert. [#]_
#
# .. [#] The most common choice of data is a `file` object with the text
#        or code source.
#
#        To convert a string into a suitable object, use its splitlines method
#        like ``"2 lines\nof source".splitlines(True)``.
#
#
# Additional keyword arguments are stored as instance variables,
# overwriting the class defaults::

    def __init__(self, data, **keyw):
        """data   --  iterable data object
                      (list, file, generator, string, ...)
           **keyw --  remaining keyword arguments are
                      stored as data-attributes
        """
        self.data = data
        self.__dict__.update(keyw)

# If empty, `code_block_marker` and `comment_string` are set according
# to the `language`::

        if not self.code_block_marker:
            self.code_block_marker = self.code_block_markers[self.language]
        if not self.comment_string:
            self.comment_string = self.comment_strings[self.language]
        self.stripped_comment_string = self.comment_string.rstrip()

# Pre- and postprocessing filters are set (with
# `TextCodeConverter.get_filter`_)::

        self.preprocessor = self.get_filter("preprocessors", self.language)
        self.postprocessor = self.get_filter("postprocessors", self.language)

# .. _inserted into a regular expression:
#
# Finally, a regular_expression for the `code_block_marker` is compiled
# to find valid cases of `code_block_marker` in a given line and return
# the groups: ``\1 prefix, \2 code_block_marker, \3 remainder`` ::

        marker = self.code_block_marker
        if marker == '::':
            # the default marker may occur at the end of a text line
            self.marker_regexp = re.compile('^( *(?!\.\.).*)(::)([ \n]*)$')
        else:
            # marker must be on a separate line
            self.marker_regexp = re.compile('^( *)(%s)(.*\n?)$' % marker)

# .. _TextCodeConverter.__iter__:
#
# __iter__
# """"""""
#
# Return an iterator for the instance. Iteration yields lines of converted
# data.
#
# The iterator is a chain of iterators acting on `self.data` that does
#
# * preprocessing
# * text<->code format conversion
# * postprocessing
#
# Pre- and postprocessing are only performed, if filters for the current
# language are registered in `defaults.preprocessors`_ and|or
# `defaults.postprocessors`_. The filters must accept an iterable as first
# argument and yield the processed input data line-wise.
# ::

    def __iter__(self):
        """Iterate over input data source and yield converted lines
        """
        return self.postprocessor(self.convert(self.preprocessor(self.data)))


# .. _TextCodeConverter.__call__:
#
# __call__
# """"""""
# The special `__call__` method allows the use of class instances as callable
# objects. It returns the converted data as list of lines::

    def __call__(self):
        """Iterate over state-machine and return results as list of lines"""
        return [line for line in self]


# .. _TextCodeConverter.__str__:
#
# __str__
# """""""
# Return converted data as string::

    def __str__(self):
        return "".join(self())


# Helpers and convenience methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. _TextCodeConverter.convert:
#
# convert
# """""""
#
# The `convert` method generates an iterator that does the actual  code <-->
# text format conversion. The converted data is yielded line-wise and the
# instance's `status` argument indicates whether the current line is "header",
# "documentation", or "code_block"::

    def convert(self, lines):
        """Iterate over lines of a program document and convert
        between "text" and "code" format
        """

# Initialise internal data arguments. (Done here, so that every new iteration
# re-initialises them.)
#
# `state`
#   the "type" of the currently processed block of lines. One of
#
#   :"":              initial state: check for header,
#   :"header":        leading code block: strip `header_string`,
#   :"documentation": documentation part: comment out,
#   :"code_block":    literal blocks containing source code: unindent.
#
# ::

        self.state = ""

# `_codeindent`
#   * Do not confuse the internal attribute `_codeindent` with the configurable
#     `codeindent` (without the leading underscore).
#   * `_codeindent` is set in `Text2Code.code_block_handler`_ to the indent of
#     first non-blank "code_block" line and stripped from all "code_block" lines
#     in the text-to-code conversion,
#   * `codeindent` is set in `__init__` to `defaults.codeindent`_ and added to
#     "code_block" lines in the code-to-text conversion.
#
# ::

        self._codeindent = 0

# `_textindent`
#   * set by `Text2Code.documentation_handler`_ to the minimal indent of a
#     documentation block,
#   * used in `Text2Code.set_state`_ to find the end of a code block.
#
# ::

        self._textindent = 0

# `_add_code_block_marker`
#   If the last paragraph of a documentation block does not end with a
#   code_block_marker_, it should be added (otherwise, the back-conversion
#   fails.).
#
#   `_add_code_block_marker` is set by `Code2Text.documentation_handler`_
#   and evaluated by `Code2Text.code_block_handler`_, because the
#   documentation_handler does not know whether the next block will be
#   documentation (with no need for a code_block_marker) or a code block.
#
# ::

        self._add_code_block_marker = False



# Determine the state of the block and convert with the matching "handler"::

        for block in collect_blocks(expandtabs_filter(lines)):
            self.set_state(block)
            for line in getattr(self, self.state+"_handler")(block):
                yield line


# .. _TextCodeConverter.get_filter:
#
# get_filter
# """"""""""
# ::

    def get_filter(self, filter_set, language):
        """Return language specific filter"""
        if self.__class__ == Text2Code:
            key = "text2"+language
        elif self.__class__ == Code2Text:
            key = language+"2text"
        else:
            key = ""
        try:
            return getattr(defaults, filter_set)[key]
        except (AttributeError, KeyError):
            # print("there is no %r filter in %r"%(key, filter_set))
            pass
        return identity_filter


# get_indent
# """"""""""
# Return the number of leading spaces in `line`::

    def get_indent(self, line):
        """Return the indentation of `string`.
        """
        return len(line) - len(line.lstrip())


# Text2Code
# ---------
#
# The `Text2Code` converter separates *code-blocks* [#]_ from *documentation*.
# Code blocks are unindented, documentation is commented (or filtered, if the
# ``strip`` option is True).
#
# .. [#] Only `indented literal blocks`_ are considered code-blocks. `quoted
#        literal blocks`_, `parsed-literal blocks`_, and `doctest blocks`_ are
#        treated as part of the documentation. This allows the inclusion of
#        examples:
#
#           >>> 23 + 3
#           26
#
#        Mark that there is no double colon before the doctest block in the
#        text source.
#
# The class inherits the interface and helper functions from
# TextCodeConverter_ and adds functions specific to the text-to-code format
# conversion::

class Text2Code(TextCodeConverter):
    """Convert a (reStructured) text source to code source
    """

# .. _Text2Code.set_state:
#
# set_state
# ~~~~~~~~~
# ::

    def set_state(self, block):
        """Determine state of `block`. Set `self.state`
        """

# `set_state` is used inside an iteration. Hence, if we are out of data, a
# StopItertion exception should be raised::

        if not block:
            raise StopIteration

# The new state depends on the active state (from the last block) and
# features of the current block. It is either "header", "documentation", or
# "code_block".
#
# If the current state is "" (first block), check for
# the  `header_string` indicating a leading code block::

        if self.state == "":
            # print("set state for %r"%block)
            if block[0].startswith(self.header_string):
                self.state = "header"
            else:
                self.state = "documentation"

# If the current state is "documentation", the next block is also
# documentation. The end of a documentation part is detected in the
# `Text2Code.documentation_handler`_::

        # elif self.state == "documentation":
        #    self.state = "documentation"

# A "code_block" ends with the first less indented, non-blank line.
# `_textindent` is set by the documentation handler to the indent of the
# preceding documentation block::

        elif self.state in ["code_block", "header"]:
            indents = [self.get_indent(line) for line in block
                       if line.rstrip()]
            # print("set_state:", indents, self._textindent)
            if indents and min(indents) <= self._textindent:
                self.state = 'documentation'
            else:
                self.state = 'code_block'

# TODO: (or not to do?) insert blank line before the first line with too-small
# codeindent using self.ensure_trailing_blank_line(lines, line) (would need
# split and push-back of the documentation part)?
#
# .. _Text2Code.header_handler:
#
# header_handler
# ~~~~~~~~~~~~~~
#
# Sometimes code needs to remain on the first line(s) of the document to be
# valid. The most common example is the "shebang" line that tells a POSIX
# shell how to process an executable file::

#!/usr/bin/env python

# In Python, the special comment to indicate the encoding, e.g.
# ``# -*- coding: iso-8859-1 -*-``, must occur before any other comment
# or code too.
#
# If we want to keep the line numbers in sync for text and code source, the
# reStructured Text markup for these header lines must start at the same line
# as the first header line. Therefore, header lines could not be marked as
# literal block (this would require the ``::`` and an empty line above the
# code_block).
#
# OTOH, a comment may start at the same line as the comment marker and it
# includes subsequent indented lines. Comments are visible in the reStructured
# Text source but hidden in the pretty-printed output.
#
# With a header converted to comment in the text source, everything before
# the first documentation block (i.e. before the first paragraph using the
# matching comment string) will be hidden away (in HTML or PDF output).
#
# This seems a good compromise, the advantages
#
# * line numbers are kept
# * the "normal" code_block conversion rules (indent/unindent by `codeindent` apply
# * greater flexibility: you can hide a repeating header in a project
#   consisting of many source files.
#
# set off the disadvantages
#
# - it may come as surprise if a part of the file is not "printed",
# - one more syntax element to learn for rst newbies to start with pylit,
#   (however, starting from the code source, this will be auto-generated)
#
# In the case that there is no matching comment at all, the complete code
# source will become a comment -- however, in this case it is not very likely
# the source is a literate document anyway.
#
# If needed for the documentation, it is possible to quote the header in (or
# after) the first documentation block, e.g. as `parsed literal`.
# ::

    def header_handler(self, lines):
        """Format leading code block"""
        # strip header string from first line
        lines[0] = lines[0].replace(self.header_string, "", 1)
        # yield remaining lines formatted as code-block
        for line in self.code_block_handler(lines):
            yield line


# .. _Text2Code.documentation_handler:
#
# documentation_handler
# ~~~~~~~~~~~~~~~~~~~~~
#
# The 'documentation' handler processes everything that is not recognised as
# "code_block". Documentation is quoted with `self.comment_string`
# (or filtered with `--strip=True`).
#
# If end-of-documentation marker is detected,
#
# * set state to 'code_block'
# * set `self._textindent` (needed by `Text2Code.set_state`_ to find the
#   next "documentation" block)
#
# ::

    def documentation_handler(self, lines):
        """Convert documentation blocks from text to code format
        """
        for line in lines:
            # test lines following the code-block marker for false positives
            if (self.state == "code_block" and line.rstrip()
                and not self.directive_option_regexp.search(line)):
                self.state = "documentation"
            # test for end of documentation block
            if self.marker_regexp.search(line):
                self.state = "code_block"
                self._textindent = self.get_indent(line)
            # yield lines
            if self.strip:
                continue
            # do not comment blank lines preceding a code block
            if self.state == "code_block" and not line.rstrip():
                yield line
            else:
                yield self.comment_string + line




# .. _Text2Code.code_block_handler:
#
# code_block_handler
# ~~~~~~~~~~~~~~~~~~
#
# The "code_block" handler is called with an indented literal block. It
# removes leading whitespace up to the indentation of the first code line in
# the file (this deviation from Docutils behaviour allows indented blocks of
# Python code). ::

    def code_block_handler(self, block):
        """Convert indented literal blocks to source code format
        """

# If still unset, determine the indentation of code blocks from first non-blank
# code line::

        if self._codeindent == 0:
            self._codeindent = self.get_indent(block[0])

# Yield unindented lines after check whether we can safely unindent. If the
# line is less indented then `_codeindent`, something got wrong. ::

        for line in block:
            if line.lstrip() and self.get_indent(line) < self._codeindent:
                raise ValueError("code block contains line less indented "
                      "than %d spaces \n%r" % (self._codeindent, block))
            yield line.replace(" "*self._codeindent, "", 1)


# Code2Text
# ---------
#
# The `Code2Text` converter does the opposite of `Text2Code`_ -- it processes
# a source in "code format" (i.e. in a programming language), extracts
# documentation from comment blocks, and puts program code in literal blocks.
#
# The class inherits the interface and helper functions from
# TextCodeConverter_ and adds functions specific to the text-to-code  format
# conversion::

class Code2Text(TextCodeConverter):
    """Convert code source to text source
    """

# set_state
# ~~~~~~~~~
#
# Check if block is "header", "documentation", or "code_block":
#
# A paragraph is "documentation", if every non-blank line starts with a
# matching comment string (including whitespace except for commented blank
# lines) ::

    def set_state(self, block):
        """Determine state of `block`."""
        for line in block:
            # skip documentation lines (commented, blank or blank comment)
            if (line.startswith(self.comment_string)
                or not line.rstrip()
                or line.rstrip() == self.comment_string.rstrip()
               ):
                continue
            # non-commented line found:
            if self.state == "":
                self.state = "header"
            else:
                self.state = "code_block"
            break
        else:
            # no code line found
            # keep state if the block is just a blank line
            # if len(block) == 1 and self._is_blank_codeline(line):
            #     return
            self.state = "documentation"


# header_handler
# ~~~~~~~~~~~~~~
#
# Handle a leading code block. (See `Text2Code.header_handler`_ for a
# discussion of the "header" state.) ::

    def header_handler(self, lines):
        """Format leading code block"""
        if self.strip == True:
            return
        # get iterator over the lines that formats them as code-block
        lines = iter(self.code_block_handler(lines))
        # prepend header string to first line
        yield self.header_string + lines.next()
        # yield remaining lines
        for line in lines:
            yield line

# .. _Code2Text.documentation_handler:
#
# documentation_handler
# ~~~~~~~~~~~~~~~~~~~~~
#
# The *documentation state* handler converts a comment to a documentation
# block by stripping the leading `comment string` from every line::

    def documentation_handler(self, block):
        """Uncomment documentation blocks in source code
        """

# Strip comment strings::

        lines = [self.uncomment_line(line) for line in block]

# If the code block is stripped, the literal marker would lead to an
# error when the text is converted with Docutils. Strip it as well. ::

        if self.strip or self.strip_marker:
            self.strip_code_block_marker(lines)

# Otherwise, check for the `code_block_marker`_ at the end of the
# documentation block (skipping directive options that might follow it)::

        elif self.add_missing_marker:
            for line in lines[::-1]:
                if self.marker_regexp.search(line):
                    self._add_code_block_marker = False
                    break
                if (line.rstrip() and
                    not self.directive_option_regexp.search(line)):
                    self._add_code_block_marker = True
                    break
            else:
                self._add_code_block_marker = True

# Yield lines::

        for line in lines:
            yield line

# uncomment_line
# ~~~~~~~~~~~~~~
#
# Return documentation line after stripping comment string. Consider the
# case that a blank line has a comment string without trailing whitespace::

    def uncomment_line(self, line):
        """Return uncommented documentation line"""
        line = line.replace(self.comment_string, "", 1)
        if line.rstrip() == self.stripped_comment_string:
            line = line.replace(self.stripped_comment_string, "", 1)
        return line

# .. _Code2Text.code_block_handler:
#
# code_block_handler
# ~~~~~~~~~~~~~~~~~~
#
# The `code_block` handler returns the code block as indented literal
# block (or filters it, if ``self.strip == True``). The amount of the code
# indentation is controlled by `self.codeindent` (default 2).  ::

    def code_block_handler(self, lines):
        """Covert code blocks to text format (indent or strip)
        """
        if self.strip == True:
            return
        # eventually insert transition marker
        if self._add_code_block_marker:
            self.state = "documentation"
            yield self.code_block_marker + "\n"
            yield "\n"
            self._add_code_block_marker = False
            self.state = "code_block"
        for line in lines:
            yield " "*self.codeindent + line



# strip_code_block_marker
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Replace the literal marker with the equivalent of Docutils replace rules
#
# * strip ``::``-line (and preceding blank line) if on a line on its own
# * strip ``::`` if it is preceded by whitespace.
# * convert ``::`` to a single colon if preceded by text
#
# `lines` is a list of documentation lines (with a trailing blank line).
# It is modified in-place::

    def strip_code_block_marker(self, lines):
        try:
            line = lines[-2]
        except IndexError:
            return # just one line (no trailing blank line)

        # match with regexp: `match` is None or has groups
        # \1 leading text, \2 code_block_marker, \3 remainder
        match = self.marker_regexp.search(line)

        if not match:                 # no code_block_marker present
            return
        if not match.group(1):        # `code_block_marker` on an extra line
            del(lines[-2])
            # delete preceding line if it is blank
            if len(lines) >= 2 and not lines[-2].lstrip():
                del(lines[-2])
        elif match.group(1).rstrip() < match.group(1):
            # '::' follows whitespace
            lines[-2] = match.group(1).rstrip() + match.group(3)
        else:                         # '::' follows text
            lines[-2] = match.group(1).rstrip() + ':' + match.group(3)

# Filters
# =======
#
# Filters allow pre- and post-processing of the data to bring it in a format
# suitable for the "normal" text<->code conversion. An example is conversion
# of `C` ``/*`` ``*/`` comments into C++ ``//`` comments (and back).
# Another example is the conversion of `C` ``/*`` ``*/`` comments into C++
# ``//`` comments (and back).
#
# Filters are generator functions that return an iterator acting on a
# `data` iterable and yielding processed `data` lines.
#
# identity_filter
# ---------------
#
# The most basic filter is the identity filter, that returns its argument as
# iterator::

def identity_filter(data):
    """Return data iterator without any processing"""
    return iter(data)

# expandtabs_filter
# -----------------
#
# Expand hard-tabs in every line of `data` (cf. `str.expandtabs`).
#
# This filter is applied to the input data by `TextCodeConverter.convert`_ as
# hard tabs can lead to errors when the indentation is changed. ::

def expandtabs_filter(data):
    """Yield data tokens with hard-tabs expanded"""
    for line in data:
        yield line.expandtabs()


# collect_blocks
# --------------
#
# A filter to aggregate "paragraphs" (blocks separated by blank
# lines). Yields lists of lines::

def collect_blocks(lines):
    """collect lines in a list

    yield list for each paragraph, i.e. block of lines separated by a
    blank line (whitespace only).

    Trailing blank lines are collected as well.
    """
    blank_line_reached = False
    block = []
    for line in lines:
        if blank_line_reached and line.rstrip():
            yield block
            blank_line_reached = False
            block = [line]
            continue
        if not line.rstrip():
            blank_line_reached = True
        block.append(line)
    yield block



# dumb_c_preprocessor
# -------------------
#
# This is a basic filter to convert `C` to `C++` comments. Works line-wise and
# only converts lines that
#
# * start with "/\* " and end with " \*/" (followed by whitespace only)
#
# A more sophisticated version would also
#
# * convert multi-line comments
#
#   + Keep indentation or strip 3 leading spaces?
#
# * account for nested comments
#
# * only convert comments that are separated from code by a blank line
#
# ::

def dumb_c_preprocessor(data):
    """change `C` ``/* `` `` */`` comments into C++ ``// `` comments"""
    comment_string = defaults.comment_strings["c++"]
    boc_string = "/* "
    eoc_string = " */"
    for line in data:
        if (line.startswith(boc_string)
            and line.rstrip().endswith(eoc_string)
           ):
            line = line.replace(boc_string, comment_string, 1)
            line = "".join(line.rsplit(eoc_string, 1))
        yield line

# Unfortunately, the `replace` method of strings does not support negative
# numbers for the `count` argument:
#
#   >>> "foo */ baz */ bar".replace(" */", "", -1) == "foo */ baz bar"
#   False
#
# However, there is the `rsplit` method, that can be used together with `join`:
#
#   >>> "".join("foo */ baz */ bar".rsplit(" */", 1)) == "foo */ baz bar"
#   True
#
# dumb_c_postprocessor
# --------------------
#
# Undo the preparations by the dumb_c_preprocessor and re-insert valid comment
# delimiters ::

def dumb_c_postprocessor(data):
    """change C++ ``// `` comments into `C` ``/* `` `` */`` comments"""
    comment_string = defaults.comment_strings["c++"]
    boc_string = "/* "
    eoc_string = " */"
    for line in data:
        if line.rstrip() == comment_string.rstrip():
            line = line.replace(comment_string, "", 1)
        elif line.startswith(comment_string):
            line = line.replace(comment_string, boc_string, 1)
            line = line.rstrip() + eoc_string + "\n"
        yield line


# register filters
# ----------------
#
# ::

defaults.preprocessors['c2text'] = dumb_c_preprocessor
defaults.preprocessors['css2text'] = dumb_c_preprocessor
defaults.postprocessors['text2c'] = dumb_c_postprocessor
defaults.postprocessors['text2css'] = dumb_c_postprocessor


# Command line use
# ================
#
# Using this script from the command line will convert a file according to its
# extension. This default can be overridden by a couple of options.
#
# Dual source handling
# --------------------
#
# How to determine which source is up-to-date?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# - set modification date of `outfile` to the one of `infile`
#
#   Points out that the source files are 'synchronised'.
#
#   * Are there problems to expect from "backdating" a file? Which?
#
#     Looking at http://www.unix.com/showthread.php?t=20526, it seems
#     perfectly legal to set `mtime` (while leaving `ctime`) as `mtime` is a
#     description of the "actuality" of the data in the file.
#
#   * Should this become a default or an option?
#
# - alternatively move input file to a backup copy (with option: `--replace`)
#
# - check modification date before overwriting
#   (with option: `--overwrite=update`)
#
# - check modification date before editing (implemented as `Jed editor`_
#   function `pylit_check()` in `pylit.sl`_)
#
# .. _Jed editor: http://www.jedsoft.org/jed/
# .. _pylit.sl: http://jedmodes.sourceforge.net/mode/pylit/
#
# Recognised Filename Extensions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Instead of defining a new extension for "pylit" literate programs,
# by default ``.txt`` will be appended for the text source and stripped by
# the conversion to the code source. I.e. for a Python program foo:
#
# * the code source is called ``foo.py``
# * the text source is called ``foo.py.txt``
# * the html rendering is called ``foo.py.html``
#
#
# OptionValues
# ------------
#
# The following class adds `as_dict`_, `complete`_ and `__getattr__`_
# methods to `optparse.Values`::

class OptionValues(optparse.Values):

# .. _OptionValues.as_dict:
#
# as_dict
# ~~~~~~~
#
# For use as keyword arguments, it is handy to have the options in a
# dictionary. `as_dict` returns a copy of the instances object dictionary::

    def as_dict(self):
        """Return options as dictionary object"""
        return self.__dict__.copy()

# .. _OptionValues.complete:
#
# complete
# ~~~~~~~~
#
# ::

    def complete(self, **keyw):
        """
        Complete the option values with keyword arguments.

        Do not overwrite existing values. Only use arguments that do not
        have a corresponding attribute in `self`,
        """
        for key in keyw:
            if not self.__dict__.__contains__(key):
                setattr(self, key, keyw[key])

# .. _OptionValues.__getattr__:
#
# __getattr__
# ~~~~~~~~~~~
#
# To replace calls using ``options.ensure_value("OPTION", None)`` with the
# more concise ``options.OPTION``, we define `__getattr__` [#]_ ::

    def __getattr__(self, name):
        """Return default value for non existing options"""
        return None


# .. [#] The special method `__getattr__` is only called when an attribute
#        look-up has not found the attribute in the usual places (i.e. it is
#        not an instance attribute nor is it found in the class tree for
#        self).
#
#
# PylitOptions
# ------------
#
# The `PylitOptions` class comprises an option parser and methods for parsing
# and completion of command line options::

class PylitOptions(object):
    """Storage and handling of command line options for pylit"""

# Instantiation
# ~~~~~~~~~~~~~
#
# ::

    def __init__(self):
        """Set up an `OptionParser` instance for pylit command line options

        """
        p = optparse.OptionParser(usage=main.__doc__, version=_version)

        # Conversion settings

        p.add_option("-c", "--code2txt", dest="txt2code", action="store_false",
                     help="convert code source to text source")
        p.add_option("-t", "--txt2code", action="store_true",
                     help="convert text source to code source")
        p.add_option("--language",
                     choices = list(defaults.languages.values()),
                     help="use LANGUAGE native comment style")
        p.add_option("--comment-string", dest="comment_string",
                     help="documentation block marker in code source "
                     "(including trailing whitespace, "
                     "default: language dependent)")
        p.add_option("-m", "--code-block-marker", dest="code_block_marker",
                     help="syntax token starting a code block. (default '::')")
        p.add_option("--codeindent", type="int",
                     help="Number of spaces to indent code blocks with "
                     "text2code (default %d)" % defaults.codeindent)

        # Output file handling

        p.add_option("--overwrite", action="store",
                     choices = ["yes", "update", "no"],
                     help="overwrite output file (default 'update')")
        p.add_option("--replace", action="store_true",
                     help="move infile to a backup copy (appending '~')")
        p.add_option("-s", "--strip", action="store_true",
                     help='"export" by stripping documentation or code')

        # Special actions

        p.add_option("-d", "--diff", action="store_true",
                     help="test for differences to existing file")
        p.add_option("--doctest", action="store_true",
                     help="run doctest.testfile() on the text version")
        p.add_option("-e", "--execute", action="store_true",
                     help="execute code (Python only)")

        self.parser = p

# .. _PylitOptions.parse_args:
#
# parse_args
# ~~~~~~~~~~
#
# The `parse_args` method calls the `optparse.OptionParser` on command
# line or provided args and returns the result as `PylitOptions.Values`
# instance. Defaults can be provided as keyword arguments::

    def parse_args(self, args=sys.argv[1:], **keyw):
        """parse command line arguments using `optparse.OptionParser`

           parse_args(args, **keyw) -> OptionValues instance

            args --  list of command line arguments.
            keyw --  keyword arguments or dictionary of option defaults
        """
        # parse arguments
        (values, args) = self.parser.parse_args(args, OptionValues(keyw))
        # Convert FILE and OUTFILE positional args to option values
        # (other positional arguments are ignored)
        try:
            values.infile = args[0]
            values.outfile = args[1]
        except IndexError:
            pass

        return values

# .. _PylitOptions.complete_values:
#
# complete_values
# ~~~~~~~~~~~~~~~
#
# Complete an OptionValues instance `values`.  Use module-level defaults and
# context information to set missing option values to sensible defaults (if
# possible) ::

    def complete_values(self, values):
        """complete option values with module and context sensible defaults

        x.complete_values(values) -> values
        values -- OptionValues instance
        """

# Complete with module-level defaults_::

        values.complete(**defaults.__dict__)

# Ensure infile is a string::

        values.ensure_value("infile", "")

# Guess conversion direction from `infile` filename::

        if values.txt2code is None:
            in_extension = os.path.splitext(values.infile)[1]
            if in_extension in values.text_extensions:
                values.txt2code = True
            elif in_extension in values.languages.keys():
                values.txt2code = False

# Auto-determine the output file name::

        values.ensure_value("outfile", self._get_outfile_name(values))

# Second try: Guess conversion direction from outfile filename::

        if values.txt2code is None:
            out_extension = os.path.splitext(values.outfile)[1]
            values.txt2code = not (out_extension in values.text_extensions)

# Set the language of the code::

        if values.txt2code is True:
            code_extension = os.path.splitext(values.outfile)[1]
        elif values.txt2code is False:
            code_extension = os.path.splitext(values.infile)[1]
        values.ensure_value("language", values.languages[code_extension])

        return values

# _get_outfile_name
# ~~~~~~~~~~~~~~~~~
#
# Construct a matching filename for the output file. The output filename is
# constructed from `infile` by the following rules:
#
# * '-' (stdin) results in '-' (stdout)
# * strip the `text_extension`_ (txt2code) or
# * add the `text_extension`_ (code2txt)
# * fallback: if no guess can be made, add ".out"
#
#   .. TODO: use values.outfile_extension if it exists?
#
# ::

    def _get_outfile_name(self, values):
        """Return a matching output filename for `infile`
        """
        # if input is stdin, default output is stdout
        if values.infile == '-':
            return '-'

        # Derive from `infile` name: strip or add text extension
        (base, ext) = os.path.splitext(values.infile)
        if ext in values.text_extensions:
            return base # strip
        if ext in values.languages.keys() or values.txt2code == False:
            return values.infile + values.text_extensions[0] # add
        # give up
        return values.infile + ".out"

# .. _PylitOptions.__call__:
#
# __call__
# ~~~~~~~~
#
# The special `__call__` method allows to use PylitOptions instances as
# *callables*: Calling an instance parses the argument list to extract option
# values and completes them based on "context-sensitive defaults".  Keyword
# arguments are passed to `PylitOptions.parse_args`_ as default values. ::

    def __call__(self, args=sys.argv[1:], **keyw):
        """parse and complete command line args return option values
        """
        values = self.parse_args(args, **keyw)
        return self.complete_values(values)



# Helper functions
# ----------------
#
# open_streams
# ~~~~~~~~~~~~
#
# Return file objects for in- and output. If the input path is missing,
# write usage and abort. (An alternative would be to use stdin as default.
# However,  this leaves the uninitiated user with a non-responding application
# if (s)he just tries the script without any arguments) ::

def open_streams(infile = '-', outfile = '-', overwrite='update', **keyw):
    """Open and return the input and output stream

    open_streams(infile, outfile) -> (in_stream, out_stream)

    in_stream   --  open(infile) or sys.stdin
    out_stream  --  open(outfile) or sys.stdout
    overwrite   --  'yes': overwrite eventually existing `outfile`,
                    'update': fail if the `outfile` is newer than `infile`,
                    'no': fail if `outfile` exists.

                    Irrelevant if `outfile` == '-'.
    """
    if not infile:
        strerror = "Missing input file name ('-' for stdin; -h for help)"
        raise IOError((2, strerror, infile))
    if infile == '-':
        in_stream = sys.stdin
    else:
        in_stream = open(infile, 'r')

    if outfile == '-':
        out_stream = sys.stdout
    elif overwrite == 'no' and os.path.exists(outfile):
        raise IOError((1, "Output file exists!", outfile))
    elif overwrite == 'update' and is_newer(outfile, infile):
        raise IOError((1, "Output file is newer than input file!", outfile))
    else:
        out_stream = open(outfile, 'w')
    return (in_stream, out_stream)

# is_newer
# ~~~~~~~~
#
# ::

def is_newer(path1, path2):
    """Check if `path1` is newer than `path2` (using mtime)

    Compare modification time of files at path1 and path2.

    Non-existing files are considered oldest: Return False if path1 does not
    exist and True if path2 does not exist.

    Return None for equal modification time. (This evaluates to False in a
    Boolean context but allows a test for equality.)

    """
    try:
        mtime1 = os.path.getmtime(path1)
    except OSError:
        mtime1 = -1
    try:
        mtime2 = os.path.getmtime(path2)
    except OSError:
        mtime2 = -1
    # print("mtime1", mtime1, path1, "\n", "mtime2", mtime2, path2)

    if mtime1 == mtime2:
        return None
    return mtime1 > mtime2


# get_converter
# ~~~~~~~~~~~~~
#
# Get an instance of the converter state machine::

def get_converter(data, txt2code=True, **keyw):
    if txt2code:
        return Text2Code(data, **keyw)
    else:
        return Code2Text(data, **keyw)


# Use cases
# ---------
#
# run_doctest
# ~~~~~~~~~~~
# ::

def run_doctest(infile="-", txt2code=True,
                globs={}, verbose=False, optionflags=0, **keyw):
    """run doctest on the text source
    """

# Allow imports from the current working dir by prepending an empty string to
# sys.path (see doc of sys.path())::

    sys.path.insert(0, '')

# Import classes from the doctest module::

    from doctest import DocTestParser, DocTestRunner

# Read in source. Make sure it is in text format, as tests in comments are not
# found by doctest::

    (data, out_stream) = open_streams(infile, "-")
    if txt2code is False:
        keyw.update({'add_missing_marker': False})
        converter = Code2Text(data, **keyw)
        docstring = str(converter)
    else:
        docstring = data.read()

# decode doc string if there is a "magic comment" in the first or second line
# (http://docs.python.org/reference/lexical_analysis.html#encoding-declarations)
# ::

    firstlines = ' '.join(docstring.splitlines()[:2])
    match = re.search('coding[=:]\s*([-\w.]+)', firstlines)
    if match:
        docencoding = match.group(1)
        docstring = docstring.decode(docencoding)

# Use the doctest Advanced API to run all doctests in the source text::

    test = DocTestParser().get_doctest(docstring, globs, name="",
                                       filename=infile, lineno=0)
    runner = DocTestRunner(verbose, optionflags)
    runner.run(test)
    runner.summarize
    # give feedback also if no failures occurred
    if not runner.failures:
        print("%d failures in %d tests"%(runner.failures, runner.tries))
    return runner.failures, runner.tries


# diff
# ~~~~
#
# ::

def diff(infile='-', outfile='-', txt2code=True, **keyw):
    """Report differences between converted infile and existing outfile

    If outfile does not exist or is '-', do a round-trip conversion and
    report differences.
    """

    import difflib

    instream = open(infile)
    # for diffing, we need a copy of the data as list::
    data = instream.readlines()
    # convert
    converter = get_converter(data, txt2code, **keyw)
    new = converter()

    if outfile != '-' and os.path.exists(outfile):
        outstream = open(outfile)
        old = outstream.readlines()
        oldname = outfile
        newname = "<conversion of %s>"%infile
    else:
        old = data
        oldname = infile
        # back-convert the output data
        converter = get_converter(new, not txt2code)
        new = converter()
        newname = "<round-conversion of %s>"%infile

    # find and print the differences
    is_different = False
    # print(type(old), old)
    # print(type(new), new)
    delta = difflib.unified_diff(old, new,
    # delta = difflib.unified_diff(["heute\n", "schon\n"], ["heute\n", "noch\n"],
                                      fromfile=oldname, tofile=newname)
    for line in delta:
        is_different = True
        print(line, end="")
    if not is_different:
        print(oldname)
        print(newname)
        print("no differences found")
    return is_different


# execute
# ~~~~~~~
#
# Works only for python code.
#
# Does not work with `eval`, as code is not just one expression. ::

def execute(infile="-", txt2code=True, **keyw):
    """Execute the input file. Convert first, if it is a text source.
    """

    data = open(infile)
    if txt2code:
        data = str(Text2Code(data, **keyw))
    # print("executing " + options.infile)
    exec(data)


# main
# ----
#
# If this script is called from the command line, the `main` function will
# convert the input (file or stdin) between text and code formats.
#
# Option default values for the conversion can be given as keyword arguments
# to `main`_.  The option defaults will be updated by command line options and
# extended with "intelligent guesses" by `PylitOptions`_ and passed on to
# helper functions and the converter instantiation.
#
# This allows easy customisation for programmatic use -- just call `main`
# with the appropriate keyword options, e.g. ``pylit.main(comment_string="## ")``
#
# ::

def main(args=sys.argv[1:], **defaults):
    """%prog [options] INFILE [OUTFILE]

    Convert between (reStructured) text source with embedded code,
    and code source with embedded documentation (comment blocks)

    The special filename '-' stands for standard in and output.
    """

# Parse and complete the options::

    options = PylitOptions()(args, **defaults)
    # print("infile", repr(options.infile))

# Special actions with early return::

    if options.doctest:
        return run_doctest(**options.as_dict())

    if options.diff:
        return diff(**options.as_dict())

    if options.execute:
        return execute(**options.as_dict())

# Open in- and output streams::

    try:
        (data, out_stream) = open_streams(**options.as_dict())
    except IOError as ex:
        print("IOError: %s %s" % (ex.filename, ex.strerror))
        sys.exit(ex.errno)

# Get a converter instance::

    converter = get_converter(data, **options.as_dict())

# Convert and write to out_stream::

    out_stream.write(str(converter))

    if out_stream is not sys.stdout:
        print("extract written to", out_stream.name)
        out_stream.close()

# If input and output are from files, set the modification time (`mtime`) of
# the output file to the one of the input file to indicate that the contained
# information is equal. [#]_ ::

        try:
            os.utime(options.outfile, (os.path.getatime(options.outfile),
                                       os.path.getmtime(options.infile))
                    )
        except OSError:
            pass

    ## print("mtime", os.path.getmtime(options.infile),  options.infile)
    ## print("mtime", os.path.getmtime(options.outfile), options.outfile)


# .. [#] Make sure the corresponding file object (here `out_stream`) is
#        closed, as otherwise the change will be overwritten when `close` is
#        called afterwards (either explicitly or at program exit).
#
#
# Rename the infile to a backup copy if ``--replace`` is set::

    if options.replace:
        os.rename(options.infile, options.infile + "~")


# Run main, if called from the command line::

if __name__ == '__main__':
    main()


# Open questions
# ==============
#
# Open questions and ideas for further development
#
# Clean code
# ----------
#
# * can we gain from using "shutils" over "os.path" and "os"?
# * use pylint or pyChecker to enforce a consistent style?
#
# Options
# -------
#
# * Use templates for the "intelligent guesses" (with Python syntax for string
#   replacement with dicts: ``"hello %(what)s" % {'what': 'world'}``)
#
# * Is it sensible to offer the `header_string` option also as command line
#   option?
#
# treatment of blank lines
# ------------------------
#
# Alternatives: Keep blank lines blank
#
# - "never" (current setting) -> "visually merges" all documentation
#    if there is no interjacent code
#
# - "always" -> disrupts documentation blocks,
#
# - "if empty" (no whitespace). Comment if there is whitespace.
#
#   This would allow non-obstructing markup but unfortunately this is (in
#   most editors) also non-visible markup.
#
# + "if double" (if there is more than one consecutive blank line)
#
#   With this handling, the "visual gap" remains in both, text and code
#   source.
#
#
# Parsing Problems
# ----------------
#
# * Ignore "matching comments" in literal strings?
#
#   Too complicated: Would need a specific detection algorithm for every
#   language that supports multi-line literal strings (C++, PHP, Python)
#
# * Warn if a comment in code will become documentation after round-trip?
#
#
# docstrings in code blocks
# -------------------------
#
# * How to handle docstrings in code blocks? (it would be nice to convert them
#   to rst-text if ``__docformat__ == restructuredtext``)
#
# TODO: Ask at Docutils users|developers
#
# Plug-ins
# --------
#
# Specify a path for user additions and plug-ins. This would require to
# convert Pylit from a pure module to a package...
#
#   6.4.3 Packages in Multiple Directories
#
#   Packages support one more special attribute, __path__. This is initialized
#   to be a list containing the name of the directory holding the package's
#   __init__.py before the code in that file is executed. This
#   variable can be modified; doing so affects future searches for modules and
#   subpackages contained in the package.
#
#   While this feature is not often needed, it can be used to extend the set
#   of modules found in a package.
#
#
# .. References
#
# .. _Docutils: http://docutils.sourceforge.net/
# .. _Sphinx: http://sphinx.pocoo.org
# .. _Pygments: http://pygments.org/
# .. _code-block directive:
#     http://docutils.sourceforge.net/sandbox/code-block-directive/
# .. _literal block:
# .. _literal blocks:
#     http://docutils.sf.net/docs/ref/rst/restructuredtext.html#literal-blocks
# .. _indented literal block:
# .. _indented literal blocks:
#     http://docutils.sf.net/docs/ref/rst/restructuredtext.html#indented-literal-blocks
# .. _quoted literal block:
# .. _quoted literal blocks:
#     http://docutils.sf.net/docs/ref/rst/restructuredtext.html#quoted-literal-blocks
# .. _parsed-literal blocks:
#     http://docutils.sf.net/docs/ref/rst/directives.html#parsed-literal-block
# .. _doctest block:
# .. _doctest blocks:
#     http://docutils.sf.net/docs/ref/rst/restructuredtext.html#doctest-blocks
#
# .. _feature request and patch by jrioux:
#     http://developer.berlios.de/feature/?func=detailfeature&feature_id=4890&group_id=7974
