TMPFILE="var.tmp"

all:
	rm -f $(TMPFILE)
	echo "COMPILER=\"$(CXX)\"" >> $(TMPFILE)
	echo "CFLAGS_DOLFIN=\"$(CXXFLAGS)\"" >> $(TMPFILE)
	echo "LIBS_DOLFIN=\"$(DOLFIN_LFLAGS) $(LFLAGS) $(XML2_LFLAGS)\"" >> $(TMPFILE)
	echo "PACKAGE=\"$(PACKAGE)\"" >> $(TMPFILE)
	echo "VERSION=\"$(VERSION)\"" >> $(TMPFILE)
	sh config.sh

clean:
	rm -f dolfin-config
