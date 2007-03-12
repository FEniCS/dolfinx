CFLAGS  = `pkg-config --cflags dolfin`
LIBS    = `pkg-config --libs dolfin` -ldolfin-modules
CXX     = `pkg-config --variable=compiler dolfin`

DEST    = dolfin-fsi-test
OBJECTS = main.o

all: $(DEST)

install:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c -Wall -g $<
