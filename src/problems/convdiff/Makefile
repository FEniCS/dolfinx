# With DOLFIN installed system-wide
#CFLAGS  = `dolfin-config --cflags`
#LFLAGS  = `dolfin-config --lflags`
#CXX     = `dolfin-config --compiler`

# For demo programs in the DOLFIN source tree
CFLAGS  = `../../../config/dolfin-config --cflags_insource`
LFLAGS  = `../../../config/dolfin-config --lflags_insource`
CXX     = `../../../config/dolfin-config --compiler`

#OPTFLAGS = -O6 -finline-functions -funroll-loops -fstrict-aliasing -felide-constructors
#OPTFLAGS = -O2
#OPTFLAGS = -p
OPTFLAGS = -g

CXXFLAGS = $(CFLAGS) $(OPTFLAGS)
DEST     = dolfin-convdiff
OBJECTS  = main.o

CXXLINK  = $(CXX) -o $@

all: $(DEST)

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(CXXLINK) $(OBJECTS) $(CXXFLAGS) $(LFLAGS)

.C.o:
	$(CXX) $(CXXFLAGS) -c $<
