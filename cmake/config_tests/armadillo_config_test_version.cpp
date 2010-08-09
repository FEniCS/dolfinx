#include "armadillo"
#include <iostream>

using namespace arma;

int main() {
  std::cout << arma_version::major << "."
	    << arma_version::minor << "."
	    << arma_version::patch;
  return 0;
}
