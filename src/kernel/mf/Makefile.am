SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_mf.h \
                          dolfin/MatrixFactory.h \
                          dolfin/StiffnessMatrix.h \
                          dolfin/LoadVector.h \
                          dolfin/MassMatrix.h \
                          dolfin/ConvectionMatrix.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LTLIBRARIES         = libdolfin-mf.la
libdolfin_mf_la_SOURCES = MatrixFactory.cpp
libdolfin_mf_la_LDFLAGS = -static
