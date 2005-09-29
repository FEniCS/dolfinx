# Swig test

DOLFIN_CFLAGS = `../../../config/dolfin-config --cflags_dolfin`
DOLFIN_LFLAGS = `../../../config/dolfin-config --libs_dolfin`
CXX = `../../../config/dolfin-config --compiler`
SWIG = swig

CPPFLAGS = -I/usr/include/python2.4 -I/usr/local/include $(DOLFIN_CFLAGS)
LFLAGS = $(DOLFIN_LFLAGS)
SWIGFLAGS = -I../../../../include

default: pydolfin.py _pydolfin.so

pydolfin.py: pydolfin.i
	$(SWIG) -python -c++ $(SWIGFLAGS) $<
	mv pydolfin_wrap.cxx pydolfin_wrap.cpp
	echo "}" >> pydolfin_wrap.cpp

_pydolfin.so: pydolfin_wrap.o
	$(CXX) -shared $< $(LFLAGS) -o $@ $(LFLAGS)

pydolfin_wrap.o: pydolfin_wrap.cpp

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $<

clean:
	rm -f pydolfin.py pydolfin.pyc _pydolfin.so pydolfin_wrap.o pydolfin_wrap.cpp *~

