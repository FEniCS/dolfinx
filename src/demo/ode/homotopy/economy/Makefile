CFLAGS  = `pkg-config --cflags dolfin`
LIBS    = `pkg-config --libs dolfin`
CXX     = `pkg-config --variable=compiler dolfin`

DEST1    = demo-test
OBJECTS1 = test.o

DEST2    = demo-ces
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
