SUBDIRS                        = dolfin
include_HEADERS                = dolfin/dolfin_quadrature.h \
                                 dolfin/GaussQuadrature.h \
                                 dolfin/LobattoQuadrature.h \
                                 dolfin/Quadrature.h \
                                 dolfin/RadauQuadrature.h \
                                 dolfin/TetrahedronMidpointQuadrature.h \
                                 dolfin/TetrahedronVertexQuadrature.h \
                                 dolfin/TriangleMidpointQuadrature.h \
                                 dolfin/TriangleVertexQuadrature.h
INCLUDES                       = @MODULE_INCLUDES@
lib_LIBRARIES                  = libdolfin-quadrature.a
libdolfin_quadrature_a_SOURCES = GaussQuadrature.cpp \
                                 LobattoQuadrature.cpp \
                                 RadauQuadrature.cpp
