# With DOLFIN installed system-wide
#CFLAGS  = `dolfin-config --cflags`
#LIBS    = `dolfin-config --libs`
#CXX     = `dolfin-config --compiler`

# For demo programs in the DOLFIN source tree
CFLAGS  = `../../config/dolfin-config --cflags_dolfin`
LIBS    = `../../config/dolfin-config --libs_dolfin`
CXX     = `../../config/dolfin-config --compiler`

DEST    = dolfin-fem
OBJECTS = main.o

all: $(DEST)

install:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
