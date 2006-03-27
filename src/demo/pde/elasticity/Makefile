CFLAGS  = `dolfin-config --cflags`
LIBS    = `dolfin-config --libs`
CXX     = `dolfin-config --compiler`
LINK    = `dolfin-config --linker`

DEST    = dolfin-elasticity
OBJECTS = main.o

all: $(DEST)

install:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(LINK) -o $@ $(OBJECTS) -gcc-name=gcc-3.4 $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) -gcc-name=gcc-3.4 $(CFLAGS) -c $<
