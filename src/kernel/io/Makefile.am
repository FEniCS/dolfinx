SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_io.h \
                          dolfin/File.h \
                          dolfin/GenericFile.h \
                          dolfin/GiDFile.h \
                          dolfin/MatlabFile.h \
                          dolfin/MFile.h \
                          dolfin/MTXFile.h \
                          dolfin/OctaveFile.h \
                          dolfin/OpenDXFile.h \
                          dolfin/PythonFile.h \
                          dolfin/VTKFile.h \
                          dolfin/XMLBLASFormData.h \
                          dolfin/XMLFile.h \
                          dolfin/XMLFiniteElementSpec.h \
                          dolfin/XMLMesh.h \
                          dolfin/XMLMatrix.h \
                          dolfin/XMLObject.h \
                          dolfin/XMLParameterList.h \
                          dolfin/XMLFunction.h \
                          dolfin/XMLVector.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LTLIBRARIES         = libdolfin-io.la
libdolfin_io_la_SOURCES = File.cpp \
                          GenericFile.cpp \
                          GiDFile.cpp \
                          MatlabFile.cpp \
                          MFile.cpp \
                          MTXFile.cpp \
                          OctaveFile.cpp \
                          OpenDXFile.cpp \
                          PythonFile.cpp \
                          VTKFile.cpp \
                          XMLBLASFormData.cpp \
                          XMLFile.cpp \
                          XMLFiniteElementSpec.cpp \
                          XMLMesh.cpp \
                          XMLMatrix.cpp \
                          XMLObject.cpp \
                          XMLParameterList.cpp \
                          XMLFunction.cpp \
                          XMLVector.cpp
libdolfin_io_la_LDFLAGS  = -static
