TMPFILE="var.tmp"

all:
	rm -f $(TMPFILE)
	echo "hej=$(DOLFIN_LFLAGS)"
	echo "COMPILER=\"$(CXX)\"" >> $(TMPFILE)
	echo "CFLAGS=\"$(CXXFLAGS)\"" >> $(TMPFILE)
	echo "LFLAGS=\"$(DOLFIN_LFLAGS) $(LFLAGS) $(XML2_LFLAGS)\"" >> $(TMPFILE)
	echo "PACKAGE=\"$(PACKAGE)\"" >> $(TMPFILE)
	echo "VERSION=\"$(VERSION)\"" >> $(TMPFILE)
	sh config.sh

clean:
	rm -f dolfin-config
