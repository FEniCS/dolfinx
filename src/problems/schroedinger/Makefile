# With DOLFIN installed system-wide
#CFLAGS  = `dolfin-config --cflags`
#LFLAGS  = `dolfin-config --lflags`

# For test problems in subdir 'problems'
CFLAGS  = `../../config/dolfin-config --cflags_insource`
LFLAGS  = `../../config/dolfin-config --lflags_insource`

CXXFLAGS = $(CFLAGS)
DEST     = dolfin-schroedinger
OBJECTS  = main.o

CXX      = c++
CXXLINK  = $(CXX) -o $@

all: $(DEST)

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	$(CXXLINK) $(OBJECTS) $(CXXFLAGS) $(LFLAGS) 

.C.o:
	$(CXX) $(CXXFLAGS) -c $<
