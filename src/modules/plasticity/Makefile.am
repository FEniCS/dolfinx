SUBDIRS                        = dolfin
nobase_include_HEADERS         = dolfin/dolfin_plasticity.h \
                                 dolfin/PlasticityModel.h \
                                 dolfin/VonMises.h \
                                 dolfin/DruckerPrager.h \
                                 dolfin/PlasticityProblem.h \
                                 dolfin/PlasticitySolver.h

INCLUDES                       = @MODULE_CFLAGS@
lib_LTLIBRARIES                  = libdolfin-plasticity.la
libdolfin_plasticity_la_SOURCES = PlasticitySolver.cpp \
                                  PlasticityModel.cpp \
                                  DruckerPrager.cpp \
                                  VonMises.cpp \
                                  PlasticityProblem.cpp
libdolfin_plasticity_la_LDFLAGS = -static
