%ignore dolfin::info(const Variable& variable, bool verbose=false);
%ignore dolfin::info(const Parameters&);
%ignore dolfin::Table::set(std::string,std::string,uint);
%rename(_info) dolfin::info;
