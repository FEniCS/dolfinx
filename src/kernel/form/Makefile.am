SUBDIRS                  = dolfin
include_HEADERS          = dolfin/dolfin_form.h \
                           dolfin/BilinearForm.h \
                           dolfin/ElementFunction.h \
                           dolfin/Form.h \
                           dolfin/FunctionList.h \
                           dolfin/FunctionPair.h \
                           dolfin/Integral.h \
                           dolfin/LinearForm.h \
                           dolfin/PDE.h \
                           dolfin/NewPDE.h \
                           dolfin/Product.h \
                           dolfin/ShapeFunction.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LIBRARIES            = libdolfin-form.a
libdolfin_form_a_SOURCES = BilinearForm.cpp \
                           ElementFunction.cpp \
                           Form.cpp \
                           FunctionList.cpp \
                           FunctionPair.cpp \
                           Integral.cpp \
                           LinearForm.cpp \
                           PDE.cpp \
                           NewPDE.cpp \
                           Product.cpp \
                           ShapeFunction.cpp
