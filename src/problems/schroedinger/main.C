#include <dolfin.h>
#include <iostream.h>
#include <unistd.h>

int main(int argc, char **argv){

	cout << "Solving the Schroedinger equation: " << flush;

        for (int i=0;i<10;i++){
          sleep(1);
          cout << "." << flush;
        }

	cout << endl << "Done... ;)" << endl;

	return 0;
}
