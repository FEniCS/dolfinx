bin_SCRIPTS = dolfin-config
CLEANFILES  = $(bin_SCRIPTS)
EXTRA_DIST  = $(bin_SCRIPTS) 

TMPFILE="var.tmp"

all: dolfin-config

dolfin-config:
	rm -f $(TMPFILE)
	echo "COMPILER=\"$(CXX)\"" >> $(TMPFILE)
	echo "CFLAGS_DOLFIN=\"$(CXXFLAGS)\"" >> $(TMPFILE)
	echo "LIBS_DOLFIN=\"$(DOLFIN_LFLAGS) $(LFLAGS) $(XML2_LFLAGS)\"" >> $(TMPFILE)
	echo "PACKAGE=\"$(PACKAGE)\"" >> $(TMPFILE)
	echo "VERSION=\"$(VERSION)\"" >> $(TMPFILE)
	echo "PREFIX=\"$(prefix)\"" >> $(TMPFILE)
	sh config.sh
