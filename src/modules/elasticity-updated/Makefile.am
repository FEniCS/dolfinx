SUBDIRS                        = dolfin
nobase_include_HEADERS         = dolfin/dolfin_elasticityupdated.h \
                                 dolfin/ElasticityUpdated.h \
                                 dolfin/ElasticityUpdatedMass.h \
                                 dolfin/ElasticityUpdatedSigma.h \
                                 dolfin/ElasticityUpdatedSolver.h
INCLUDES                       = @MODULE_CFLAGS@
lib_LTLIBRARIES                  = libdolfin-elasticity-updated.la
libdolfin_elasticity_updated_la_SOURCES = ElasticityUpdatedSolver.cpp
libdolfin_elasticity_updated_la_LDFLAGS = -static
