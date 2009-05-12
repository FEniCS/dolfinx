%rename (__int__) dolfin::NewParameter::operator int() const;
%rename (__float__) dolfin::NewParameter::operator double() const;
%rename (to_str) dolfin::NewParameter::operator std::string() const;
