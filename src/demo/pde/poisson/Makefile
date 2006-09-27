CFLAGS  = `pkg-config dolfin --cflags`
LIBS    = `pkg-config dolfin --libs`
CXX     = g++
LINK    = g++

DEST    = dolfin-poisson
OBJECTS = main.o

all: $(DEST)

install:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(LINK) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
