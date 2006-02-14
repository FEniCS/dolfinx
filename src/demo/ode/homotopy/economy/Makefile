CFLAGS  = `dolfin-config --cflags`
LIBS    = `dolfin-config --libs`
CXX     = `dolfin-config --compiler`
LINK    = `dolfin-config --linker`

DEST1    = dolfin-ode-homotopy-test
OBJECTS1 = test.o

DEST2    = dolfin-ode-homotopy-ces
OBJECTS2 = ces.o

all: $(DEST1) $(DEST2)

install:	

clean:
	-rm -f *.o core *.core $(OBJECTS1) $(OBJECTS2) $(DEST1) $(DEST2)

$(DEST1): $(OBJECTS1)
	 $(LINK) -o $@ $(OBJECTS1) $(CFLAGS) $(LIBS)

$(DEST2): $(OBJECTS2)
	 $(LINK) -o $@ $(OBJECTS2) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
