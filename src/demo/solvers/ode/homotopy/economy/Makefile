# With DOLFIN installed system-wide
#CFLAGS  = `dolfin-config --cflags`
#LIBS    = `dolfin-config --libs`
#CXX     = `dolfin-config --compiler`

# For demo programs in the DOLFIN source tree
CFLAGS  = `../../../../../config/dolfin-config --cflags_dolfin`
LIBS    = `../../../../../config/dolfin-config --libs_dolfin`
CXX     = `../../../../../config/dolfin-config --compiler`

DEST1    = dolfin-ode-homotopy-test
OBJECTS1 = test.o

DEST2    = dolfin-ode-homotopy-ces
OBJECTS2 = ces.o

all: $(DEST1) $(DEST2)

install:	

clean:
	-rm -f *.o core *.core $(OBJECTS1) $(OBJECTS2) $(DEST1) $(DEST2)

$(DEST1): $(OBJECTS1)
	 $(CXX) -o $@ $(OBJECTS1) $(CFLAGS) $(LIBS)

$(DEST2): $(OBJECTS2)
	 $(CXX) -o $@ $(OBJECTS2) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
