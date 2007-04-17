// Things to ignore for PyDOLFIN

%ignore *::operator=;
%ignore *::operator[];
%ignore *::operator++;
%ignore operator<<;
%ignore operator<< <Mat>;
%ignore operator int;
%ignore operator dolfin::uint;
%ignore operator dolfin::real;
%ignore operator std::string;
%ignore operator bool;
%ignore *::defined;
