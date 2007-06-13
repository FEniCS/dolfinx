SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_io.h \
                          dolfin/File.h \
                          dolfin/GenericFile.h \
                          dolfin/MatlabFile.h \
                          dolfin/MFile.h \
                          dolfin/OctaveFile.h \
                          dolfin/OpenDXFile.h \
                          dolfin/PythonFile.h \
                          dolfin/VTKFile.h \
                          dolfin/XMLBLASFormData.h \
                          dolfin/XMLDofMap.h \
                          dolfin/XMLFile.h \
                          dolfin/XMLFiniteElement.h \
                          dolfin/XMLMesh.h \
                          dolfin/XMLMeshFunction.h \
                          dolfin/XMLMatrix.h \
                          dolfin/XMLObject.h \
                          dolfin/XMLParameterList.h \
                          dolfin/XMLFunction.h \
                          dolfin/XMLVector.h \
                          dolfin/XMLGraph.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LTLIBRARIES         = libdolfin-io.la
libdolfin_io_la_SOURCES = File.cpp \
                          GenericFile.cpp \
                          MatlabFile.cpp \
                          MFile.cpp \
                          OctaveFile.cpp \
                          OpenDXFile.cpp \
                          PythonFile.cpp \
                          VTKFile.cpp \
                          XMLBLASFormData.cpp \
                          XMLDofMap.cpp \
                          XMLFile.cpp \
                          XMLFiniteElement.cpp \
                          XMLMesh.cpp \
                          XMLMeshFunction.cpp \
                          XMLMatrix.cpp \
                          XMLObject.cpp \
                          XMLParameterList.cpp \
                          XMLFunction.cpp \
                          XMLVector.cpp \
                          XMLGraph.cpp
libdolfin_io_la_LDFLAGS  = -static
