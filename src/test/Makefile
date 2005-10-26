CFLAGS  = `../config/dolfin-config --cflags_dolfin`
LIBS    = `../config/dolfin-config --libs_dolfin`
CXX     = `../config/dolfin-config --compiler`

DEST    = dolfin-test
OBJECTS = main.o

all: $(DEST)

install:

uninstall:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
