TMPFILE="var.tmp"

all:
	rm -f $(TMPFILE)
	echo "hej=$(DOLFIN_LFLAGS)"
	echo "COMPILER=\"$(CXX)\"" >> $(TMPFILE)
	echo "CFLAGS=\"$(CFLAGS)\"" >> $(TMPFILE)
	echo "LFLAGS=\"$(DOLFIN_LFLAGS) $(LFLAGS)\"" >> $(TMPFILE)
	echo "PACKAGE=\"$(PACKAGE)\"" >> $(TMPFILE)
	echo "VERSION=\"$(VERSION)\"" >> $(TMPFILE)
	sh config.sh

clean:
	rm -f dolfin-config
