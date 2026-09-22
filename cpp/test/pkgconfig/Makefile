CXX      = $(shell pkg-config --variable compiler dolfinx)
CXXFLAGS = $(shell pkg-config --cflags dolfinx)
# --static pulls in Libs.private/Requires.private too (e.g. SCOTCH,
# ParMETIS, KaHIP, SuperLU_DIST): required for a static dolfinx, and a
# no-op for a shared one, since those fields are then empty.
LDLIBS   = $(shell pkg-config --static --libs dolfinx)
LDFLAGS  = -Wl,-rpath,$(shell pkg-config --variable=libdir dolfinx)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main: main.o
	$(CXX) $(LDFLAGS) $< -o $@ $(LDLIBS)

run: main
	mpiexec -np 1 ./main

clean:
	rm -f main main.o

.PHONY: run clean
