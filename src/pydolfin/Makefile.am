BUILT_SOURCES = dolfin_wrap.cpp dolfin.py
SWIG_SOURCES = dolfin.i

pkgpyexecdir = $(pyexecdir)/dolfin
pkgpythondir = $(pyexecdir)/dolfin
pkgpython_PYTHON = dolfin.py plot.py assemble.py __init__.py
pkgpyexec_LTLIBRARIES = _dolfin.la
_dolfin_la_SOURCES = dolfin_wrap.cpp $(SWIG_SOURCES)
_dolfin_la_CPPFLAGS = @SWIG_PYTHON_CPPFLAGS@ @DOLFIN_CFLAGS@ -I$(top_builddir)/src/pydolfin
_dolfin_la_LDFLAGS = -module
_dolfin_la_LIBADD = $(top_builddir)/src/lib/libdolfin.la

# Need -fno-strict-aliasing to avoid "dereferencing type-punned pointer" warning
# Need -Wno-uninitialized avoid spurious error under Cygwin
CXXFLAGS = @DOLFIN_DEFINITIONS@ -g -O2 -fno-strict-aliasing -Wall -Werror -ansi -Wno-uninitialized

# Add include path to ufc.h
UFC_PATH = `pkg-config --cflags ufc-1` -I/usr/include

dolfin_wrap.cpp : $(SWIG_SOURCES)
	$(SWIG) $(SWIG_PYTHON_OPT) -ignoremissing -nodefaultctor -nodefaultdtor -dirprot -Werror $(UFC_PATH) @KERNEL_INCLUDE@ @DOLFIN_DEFINITIONS@ -o $@ $<

dolfin.py: $(SWIG_SOURCES)
