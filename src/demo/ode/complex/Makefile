CFLAGS  = `dolfin-config --cflags`
LIBS    = `dolfin-config --libs`
CXX     = `dolfin-config --compiler`

DEST    = dolfin-ode-complex
OBJECTS = main.o

all: $(DEST)

install:	

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
