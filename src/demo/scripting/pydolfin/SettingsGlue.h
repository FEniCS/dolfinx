#include <dolfin.h>

void odeinit(std::string method, int order);
void glueset(std::string name, dolfin::real val);
void glueset(std::string name, int val);
void glueset(std::string name, bool val);
void glueset(std::string name, std::string val);
