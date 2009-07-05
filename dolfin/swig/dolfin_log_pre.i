%ignore dolfin::info(const Variable&);
%ignore dolfin::info(const Parameters&);
%ignore dolfin::Table::set(std::string,std::string,uint);
%rename(_info) dolfin::info;
