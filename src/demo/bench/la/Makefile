OPTFLAGS = -O6 -finline-functions -funroll-loops -fstrict-aliasing -felide-constructors
CFLAGS   = -I../../../../include $(OPTFLAGS)
LFLAGS   = -L../../../../lib -ldolfin-la -ldolfin-common
CXX      = g++
DEST1    = bench-dolfin
DEST2    = bench-mtl
DEST3    = bench-test
OBJECTS1 = bench-dolfin.o
OBJECTS2 = bench-mtl.o
OBJECTS3 = bench-test.o

CXXLINK = $(CXX) -o $@

all: $(DEST1) $(DEST2)

clean:
	-rm -f *.o core *.core $(OBJECTS1) $(OBJECTS2) $(DEST1) $(DEST2)

$(DEST1): $(OBJECTS1)
	$(CXXLINK) -o $(DEST1) $(OBJECTS1) $(CXXFLAGS) $(LFLAGS)

$(DEST2): $(OBJECTS2)
	$(CXXLINK) -o $(DEST2) $(OBJECTS2) $(CXXFLAGS) $(LFLAGS)

$(DEST3): $(OBJECTS3)
	$(CXXLINK) -o $(DEST3) $(OBJECTS3) $(CXXFLAGS) $(LFLAGS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
