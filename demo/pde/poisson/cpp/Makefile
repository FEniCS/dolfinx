CFLAGS  = `pkg-config --cflags dolfin`
LIBS    = `pkg-config --libs dolfin`
CXX     = `pkg-config --variable=compiler dolfin`

DEST    = demo
OBJECTS = main.o

all: 	$(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

clean:	-rm -f $(OBJECTS) $(DEST)

.cpp.o:	$(CXX) $(CFLAGS) -c $<
