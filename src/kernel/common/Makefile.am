SUBDIRS = dolfin

include_HEADERS = dolfin/dolfin_common.h \
                  dolfin/Array.h \
                  dolfin/NewArray.h \
                  dolfin/BCFunctionPointer.h \
                  dolfin/FunctionPointer.h \
                  dolfin/General.h \
                  dolfin/IndexPair.h \
                  dolfin/List.h \
                  dolfin/NewList.h \
                  dolfin/System.h \
                  dolfin/Table.h \
                  dolfin/Tensor.h \
                  dolfin/Variable.h \
                  dolfin/VFunctionPointer.h \
                  dolfin/meminfo.h \
                  dolfin/sysinfo.h \
                  dolfin/timeinfo.h \
                  dolfin/utils.h

INCLUDES = @MODULE_INCLUDES@

lib_LIBRARIES = libdolfin-common.a

libdolfin_common_a_SOURCES = IndexPair.cpp \
                             System.cpp \
                             Variable.cpp \
                             meminfo.cpp \
                             sysinfo.cpp \
                             timeinfo.cpp \
                             utils.cpp
