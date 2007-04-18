BUILT_SOURCES = dolfin_wrap.cpp dolfin.py
SWIG_SOURCES = dolfin.i

pkgpyexecdir = $(pyexecdir)/dolfin
pkgpythondir = $(pyexecdir)/dolfin
pkgpython_PYTHON = dolfin.py import_form.py assemble.py projection.py plot.py __init__.py
pkgpyexec_LTLIBRARIES = _dolfin.la
_dolfin_la_SOURCES = dolfin_wrap.cpp dolfin_glue.cpp $(SWIG_SOURCES)
_dolfin_la_CPPFLAGS = @SWIG_PYTHON_CPPFLAGS@ @DOLFIN_CFLAGS@ -I$(top_builddir)/src/pydolfin
_dolfin_la_LDFLAGS = -module
_dolfin_la_LIBADD = $(top_builddir)/src/lib/libdolfin.la

# Need -fno-strict-aliasing to avoid "dereferencing type-punned pointer" warning
CXXFLAGS = @DOLFIN_DEFINITIONS@ -g -O2 -fno-strict-aliasing -Wall -Werror -ansi

# Temporary fix: add elasticity-updated to include path manually
#dolfin_wrap.cpp : $(SWIG_SOURCES)
#	$(SWIG) $(SWIG_PYTHON_OPT) -includeall -ignoremissing -nodefaultctor -nodefaultdtor -dirprot @KERNEL_INCLUDE@ @DOLFIN_DEFINITIONS@ -o $@ $<
dolfin_wrap.cpp : $(SWIG_SOURCES)
	$(SWIG) $(SWIG_PYTHON_OPT) -ignoremissing -nodefaultctor -nodefaultdtor -dirprot @KERNEL_INCLUDE@ @DOLFIN_DEFINITIONS@ -o $@ $<

dolfin.py: $(SWIG_SOURCES)
