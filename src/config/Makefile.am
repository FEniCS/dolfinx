bin_SCRIPTS = dolfin-config dolfin-swig
CLEANFILES  = $(bin_SCRIPTS)
EXTRA_DIST  = $(bin_SCRIPTS) 

TMPFILE="var.tmp"

all: dolfin-config dolfin-swig

dolfin-config:
	rm -f $(TMPFILE)
	echo "COMPILER=\"libtool --silent --mode=compile --tag CXX $(CXX)\"" >> $(TMPFILE)
	echo "LINKER=\"libtool --silent --mode=link --tag CXX $(CXX)\"" >> $(TMPFILE)
	echo "CFLAGS=\"$(DOLFIN_CFLAGS) $(CXXFLAGS)\"" >> $(TMPFILE)
	echo "LIBS=\"$(DOLFIN_LIBS)\"" >> $(TMPFILE)
	echo "PACKAGE=\"$(PACKAGE_NAME)\"" >> $(TMPFILE)
	echo "VERSION=\"$(PACKAGE_VERSION)\"" >> $(TMPFILE)
	echo "PREFIX=\"$(prefix)\"" >> $(TMPFILE)
	echo "PYTHON_CPPFLAGS=\"@PYTHON_CPPFLAGS@\"" >> $(TMPFILE)
	sh config.sh

dolfin-swig:
	sh swig.sh
