// Renames for PyDOLFIN

//%rename(__repr__) *::operator<<;

%rename(__setitem__) *::setitem;
%rename(__getitem__) *::getitem;
%rename(__float__) *::operator real;
