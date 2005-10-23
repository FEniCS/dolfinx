# Swig test

DOLFIN_CFLAGS = `../../../config/dolfin-config --cflags_dolfin`
DOLFIN_LFLAGS = `../../../config/dolfin-config --libs_dolfin`
CXX = `../../../config/dolfin-config --compiler`
SWIG = swig

CPPFLAGS = -I/usr/include/python2.4 $(DOLFIN_CFLAGS) -I. -g -fPIC
LFLAGS = $(DOLFIN_LFLAGS)
SWIGFLAGS = -I../../../../include

default: all

all: dolfin/dolfin.py dolfin/_dolfin.so

dolfin/dolfin.py: dolfin.i
	$(SWIG) -python -c++ -nodefault -dirprot $(SWIGFLAGS) -outdir dolfin $<
	mv dolfin_wrap.cxx dolfin_wrap.cpp

dolfin/_dolfin.so: dolfin_wrap.o PoissonTest.o SettingsGlue.o
	$(CXX) -g -shared dolfin_wrap.o PoissonTest.o SettingsGlue.o -o $@ $(LFLAGS) 

dolfin_wrap.o: dolfin_wrap.cpp

clean:
	rm -f dolfin/dolfin.py dolfin.pyc dolfin/_dolfin.so dolfin_wrap.o dolfin_wrap.cpp SettingsGlue.o PoissonTest.o *~

