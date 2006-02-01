SUBDIRS                     = dolfin
nobase_include_HEADERS      = dolfin/dolfin_heat.h \
                              dolfin/Heat.h \
                              dolfin/HeatSolver.h
INCLUDES                    = @MODULE_CFLAGS@
lib_LTLIBRARIES               = libdolfin-heat.la
libdolfin_heat_la_SOURCES = HeatSolver.cpp
libdolfin_heat_la_LDFLAGS = -static
