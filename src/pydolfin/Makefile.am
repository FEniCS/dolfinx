BUILT_SOURCES = dolfin_wrap.cpp dolfin.py
SWIG_SOURCES = dolfin.i

pkgpyexecdir = $(pyexecdir)/dolfin
pkgpythondir = $(pyexecdir)/dolfin
pkgpython_PYTHON = dolfin.py import_form.py assemble.py __init__.py
pkgpyexec_LTLIBRARIES = _dolfin.la
_dolfin_la_SOURCES = dolfin_wrap.cpp dolfin_glue.cpp $(SWIG_SOURCES)
_dolfin_la_CPPFLAGS = @SWIG_PYTHON_CPPFLAGS@ @DOLFIN_CFLAGS@ -I$(top_builddir)/src/pydolfin
_dolfin_la_LDFLAGS = -module @DOLFIN_EXT_LIBS@
_dolfin_la_LIBADD = $(top_builddir)/src/lib/libdolfin.la

# Temporary fix: don't include -Werr to get around problem with Numeric/arrayobject.h
CXXFLAGS = @DOLFIN_DEFINITIONS@ -g -O2 -Wall -ansi

dolfin_wrap.cpp : $(SWIG_SOURCES)
	echo $(SWIG_PYTHON_CPPFLAGS)
	echo $(DOLFIN_CFLAGS)
	$(SWIG) $(SWIG_PYTHON_OPT) -nodefault -dirprot @DOLFIN_CFLAGS@ @DOLFIN_DEFINITIONS@ -o $@ $<

dolfin.py: $(SWIG_SOURCES)
