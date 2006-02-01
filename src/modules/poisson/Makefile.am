SUBDIRS                     = dolfin
nobase_include_HEADERS      = dolfin/dolfin_poisson.h \
                              dolfin/Poisson2D.h \
                              dolfin/Poisson3D.h \
                              dolfin/PoissonSolver.h
INCLUDES                    = @MODULE_CFLAGS@
lib_LTLIBRARIES               = libdolfin-poisson.la
libdolfin_poisson_la_SOURCES = PoissonSolver.cpp
libdolfin_poisson_la_LDFLAGS = -static
