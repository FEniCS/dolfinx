SUBDIRS                     = dolfin
nobase_include_HEADERS      = dolfin/dolfin_ale.h \
                              dolfin/ALEBoundaryCondition.h \
                              dolfin/ALEContinuity2D.h \
                              dolfin/ALEContinuity3D.h \
                              dolfin/ALEFunction.h \
                              dolfin/ALEMomentum2D.h \
                              dolfin/ALEMomentum3D.h \
                              dolfin/ALESolver.h
INCLUDES                    = @MODULE_CFLAGS@
lib_LTLIBRARIES               = libdolfin-ale.la
libdolfin_ale_la_SOURCES = ALESolver.cpp ALEBoundaryCondition.cpp
libdolfin_ale_la_LDFLAGS = -static
