CFLAGS  = `pkg-config dolfin --cflags` -I/usr/local/garth/software/ParMetis-3.1  -I/usr/local/garth/software/ParMetis-3.1/METISLib -DMPICH_IGNORE_CXX_SEEK
LIBS    = `pkg-config dolfin --libs` -L/usr/local/garth/software/ParMetis-3.1 -lparmetis -lmetis 
CXX     = g++
LINK    = libtool --silent --mode=link --tag CXX g++

DEST    = dolfin-parallel-test
OBJECTS = main.o

all: $(DEST)

install:

uninstall:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(LINK) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS) 

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
