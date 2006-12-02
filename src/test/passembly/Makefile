PARMETIS = /usr/local/garth/software/ParMetis-3.1
#PARMETIS = /usr/src/ParMetis-3.1

CFLAGS  = `pkg-config dolfin --cflags` -I$(PARMETIS) -I$(PARMETIS)/METISLib -DMPICH_IGNORE_CXX_SEEK
LIBS    = `pkg-config dolfin --libs` -L$(PARMETIS) -lparmetis -lmetis 
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
