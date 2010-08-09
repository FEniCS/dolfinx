#include <CGAL/version.h>
#include <iostream>

int main() {
  #ifdef CGAL_VERSION_NR
    std::cout << CGAL_VERSION_NR;
  #endif
  return 0;
}
