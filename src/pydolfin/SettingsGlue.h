#include <dolfin.h>

void glueset(std::string name, dolfin::real val);
void glueset(std::string name, int val);
void glueset(std::string name, bool val);
void glueset(std::string name, std::string val);

dolfin::Parameter glueget(std::string name);
