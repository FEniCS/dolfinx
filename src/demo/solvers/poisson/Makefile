# With DOLFIN installed system-wide
#CXXFLAGS = `dolfin-config --cflags`
#LFLAGS   = `dolfin-config --lflags`
#CXX      = `dolfin-config --compiler`

# For demo programs in the DOLFIN source tree
CXXFLAGS = `../../../config/dolfin-config --cflags_insource`
LFLAGS   = `../../../config/dolfin-config --lflags_insource`
CXX      = `../../../config/dolfin-config --compiler`

DEST     = dolfin-poisson
OBJECTS  = main.o

CXXLINK  = $(CXX) -o $@

all: $(DEST)

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(CXXLINK) $(OBJECTS) $(CXXFLAGS) $(LFLAGS)
