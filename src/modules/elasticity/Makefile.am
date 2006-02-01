SUBDIRS                        = dolfin
nobase_include_HEADERS         = dolfin/dolfin_elasticity.h \
                                 dolfin/Elasticity.h \
                                 dolfin/ElasticitySolver.h
INCLUDES                       = @MODULE_CFLAGS@
lib_LTLIBRARIES                  = libdolfin-elasticity.la
libdolfin_elasticity_la_SOURCES = ElasticitySolver.cpp
libdolfin_elasticity_la_LDFLAGS = -static
