BUILT_SOURCES = dolfinmodules_wrap.cpp dolfinmodules.py
SWIG_SOURCES = dolfinmodules.i

pkgpyexecdir = $(pyexecdir)/dolfinmodules
pkgpythondir = $(pyexecdir)/dolfinmodules
pkgpython_PYTHON = dolfinmodules.py __init__.py
pkgpyexec_LTLIBRARIES = _dolfinmodules.la
_dolfinmodules_la_SOURCES = dolfinmodules_wrap.cpp $(SWIG_SOURCES)
_dolfinmodules_la_CPPFLAGS = @SWIG_PYTHON_CPPFLAGS@ @DOLFIN_CFLAGS@ @MODULE_CFLAGS@ -I$(top_builddir)/src/modules/python
_dolfinmodules_la_LDFLAGS = -module @DOLFIN_EXT_LIBS@
_dolfinmodules_la_LIBADD = $(top_builddir)/src/lib/libdolfin.la $(top_builddir)/src/modules/lib/libdolfin-modules.la

# Temporary fix: don't include -Werr to get around problem with Numeric/arrayobject.h
CXXFLAGS = @DOLFIN_DEFINITIONS@ -g -O2 -Wall -ansi

# Temporary fix: add elasticity-updated to include path manually
dolfinmodules_wrap.cpp : $(SWIG_SOURCES)
	$(SWIG) $(SWIG_PYTHON_OPT) -importall -ignoremissing -nodefault -dirprot @KERNEL_INCLUDE@ @DOLFIN_DEFINITIONS@ -o $@ $<

dolfinmodules.py: $(SWIG_SOURCES)
