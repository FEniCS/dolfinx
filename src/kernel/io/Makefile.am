SUBDIRS                = dolfin
include_HEADERS        = dolfin/dolfin_io.h \
                         dolfin/File.h \
                         dolfin/GenericFile.h \
                         dolfin/MatlabFile.h \
                         dolfin/MFile.h \
                         dolfin/OctaveFile.h \
                         dolfin/OpenDXFile.h \
                         dolfin/XMLFile.h \
                         dolfin/XMLMesh.h \
                         dolfin/XMLMatrix.h \
                         dolfin/XMLObject.h \
                         dolfin/XMLVector.h
INCLUDES               = @MODULE_INCLUDES@ @XML2_INCLUDES@
lib_LIBRARIES          = libdolfin-io.a
libdolfin_io_a_SOURCES = File.cpp \
                         GenericFile.cpp \
                         MatlabFile.cpp \
                         MFile.cpp \
                         OctaveFile.cpp \
                         OpenDXFile.cpp \
                         XMLFile.cpp \
                         XMLMesh.cpp \
                         XMLMatrix.cpp \
                         XMLObject.cpp \
                         XMLVector.cpp
