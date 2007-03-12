CFLAGS  = `pkg-config --cflags dolfin`
LIBS    = `pkg-config --libs dolfin`
CXX     = `pkg-config --variable=compiler dolfin`

DEST    = dolfin-cns
#DEST    = zero_delta/dolfin-cns-d0
#DEST    = run_delta_old_nu/dolfin-cns-dn

OBJECTS = main.o CNSSolver.o

all: $(DEST)

install:

clean:
	-rm -f *.o core *.core *.vtu *.pvd $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	 $(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
