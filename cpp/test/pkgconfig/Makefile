CXX      = mpicxx
CXXFLAGS = $(shell pkg-config --cflags dolfinx)
LDLIBS   = $(shell pkg-config --libs dolfinx)
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
