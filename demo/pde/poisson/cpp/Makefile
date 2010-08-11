CFLAGS  = $(shell pkg-config --cflags dolfin)
LIBS    = $(shell pkg-config --libs dolfin)
CXX     = $(shell pkg-config --variable=compiler dolfin)

DEST    = demo
OBJECTS = main.o

all: 	$(OBJECTS)
	$(CXX) -o $(DEST) $(OBJECTS) $(CFLAGS) $(LIBS)

clean:
	-rm -f $(OBJECTS) $(DEST)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
