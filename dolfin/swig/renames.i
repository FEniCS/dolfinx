// Renames for PyDOLFIN

//%rename(__repr__) *::operator<<;

%rename(__setitem__) *::setval;
%rename(__getitem__) dolfin::GenericMatrix::getval;
