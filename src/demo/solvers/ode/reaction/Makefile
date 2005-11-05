CFLAGS  = `dolfin-config --cflags`
LIBS    = `dolfin-config --libs`
CXX     = `dolfin-config --compiler`

DEST1    = dolfin-ode-reaction
OBJECTS1 = main.o

DEST2    = dolfin-ode-reaction-fixedpoint
OBJECTS2 = main-fixedpoint.o

all: $(DEST1) $(DEST2)

install:	

clean:
	-rm -f *.o core *.core $(OBJECTS1) $(DEST1) $(OBJECTS2) $(DEST2)

$(DEST1): $(OBJECTS1)
	 $(CXX) -o $@ $(OBJECTS1) $(CFLAGS) $(LIBS)

$(DEST2): $(OBJECTS2)
	 $(CXX) -o $@ $(OBJECTS2) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
