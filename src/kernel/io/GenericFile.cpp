#include "GenericFile.h"

// FIXME, this should not be here
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(const std::string filename)
{
  this->filename = filename;

  infile_opened = false;
  outfile_opened = false;
}
//-----------------------------------------------------------------------------
GenericFile::~GenericFile()
{
  if ( in )
	 in.close();

  if ( out )
	 out.close();
}
//-----------------------------------------------------------------------------
void GenericFile::openIn()
{
  // Check that file has not been used previously for output
  if ( outfile_opened ){
	 cout << "You cannot use the same file for input and output." << endl;
	 exit(1);
  }
  
  // Open file and check status
  in.open(filename.c_str(),ios::in);
  if ( !in ){
	 cout << "Unable to read from file \"" << filename << "\"." << endl;
	 exit(1);
  }

  // Remember that we have opened the file for input
  infile_opened = true;
}
//-----------------------------------------------------------------------------
void GenericFile::openOut()
{
  // Check that file has not been used previously for input
  if ( infile_opened ){
	 cout << "You cannot use the same file for input and output." << endl;
	 exit(1);
  }
	 
  // Open file (append if used before)
  if ( outfile_opened )
	 out.open(filename.c_str(),ios::out);
  else
	 out.open(filename.c_str(),ios::app);

  // Check file
  if ( !out ){
	 cout << "Unable to write to file \"" << filename << "\"." << endl;
	 exit(1);
  }

  // Remember that we have opened the file for output
  outfile_opened;
}
//-----------------------------------------------------------------------------
void GenericFile::closeIn()
{
  if ( !in ){
	 cout << "Internal error in GenericFile::closeIn()" << endl;
	 exit(1);
  }

  in.close();
}
//-----------------------------------------------------------------------------
void GenericFile::closeOut()
{
  if ( !out ){
	 cout << "Internal error in GenericFile::closeIn()" << endl;
	 exit(1);
  }

  out.close();
}
//-----------------------------------------------------------------------------
