CXX           = $(shell pkg-config --variable compiler dolfinx)
CXXFLAGS      = $(shell pkg-config --cflags dolfinx)
LDFLAGS       = -Wl,-rpath,$(shell pkg-config --variable=libdir dolfinx)
LDLIBS_SHARED = $(shell pkg-config --libs dolfinx)
# --static also pulls in Libs.private/Requires.private (e.g. SCOTCH,
# ParMETIS, KaHIP, SuperLU_DIST), needed only for a static dolfinx.
LDLIBS_STATIC = $(shell pkg-config --static --libs dolfinx)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main-shared: main.o
	$(CXX) $(LDFLAGS) $< -o $@ $(LDLIBS_SHARED)

main-static: main.o
	$(CXX) $(LDFLAGS) $< -o $@ $(LDLIBS_STATIC)

run-shared: main-shared
	mpiexec -np 1 ./main-shared

run-static: main-static
	mpiexec -np 1 ./main-static

clean:
	rm -f main-shared main-static main.o

.PHONY: run-shared run-static clean
