CXX      = mpicxx
CXXFLAGS = $(shell pkg-config --cflags dolfinx)
LDFLAGS  = $(shell pkg-config --libs dolfinx) -Wl,-rpath,$(shell pkg-config --variable=libdir dolfinx)

main: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: main
	mpiexec -np 1 ./main

clean:
	rm -f main

.PHONY: run clean
