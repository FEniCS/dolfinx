CFLAGS  = `pkg-config dolfin --cflags`
LIBS    = `pkg-config dolfin --libs`
CXX     = g++
LINK    = libtool --silent --mode=link --tag CXX g++

DEST    = dolfin-test
OBJECTS = main.o

all: $(DEST)

install:

uninstall:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(LINK) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS) 

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
