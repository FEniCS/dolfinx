BUILT_SOURCES = dolfin_wrap.cpp
SWIG_SOURCES = dolfin.i

pkgpyexecdir = $(pyexecdir)/dolfin
pkgpythondir = $(pyexecdir)/dolfin
pkgpython_PYTHON = dolfin.py import_form.py __init__.py
pkgpyexec_LTLIBRARIES = _dolfin.la
_dolfin_la_SOURCES = dolfin_wrap.cpp SettingsGlue.cpp PoissonTest.cpp $(SWIG_SOURCES)
_dolfin_la_CPPFLAGS = @SWIG_PYTHON_CPPFLAGS@ @DOLFIN_CFLAGS@ -I$(top_builddir)/src/pydolfin
_dolfin_la_LDFLAGS = -module @DOLFIN_EXT_LIBS@
_dolfin_la_LIBADD = $(top_builddir)/src/lib/libdolfin.la

dolfin_wrap.cpp : $(SWIG_SOURCES)
	$(SWIG) $(SWIG_PYTHON_OPT) -nodefault -dirprot @DOLFIN_CFLAGS@ -o $@ $<
