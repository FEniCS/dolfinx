#include "DolfinFile.h"

#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
DolfinFile::DolfinFile(const std::string filename) : GenericFile(filename)
{
  no_objects = 0;
  objects = 0;
}
//-----------------------------------------------------------------------------
DolfinFile::~DolfinFile()
{
  if ( objects )
	 delete [] objects;
}
//-----------------------------------------------------------------------------
void DolfinFile::operator>> (Vector& vector)
{
  cout << "Loading vector..." << endl;

  // Open file
  openIn();

  // Create index
  createIndex();
  
  // Find keywords "DOLFIN Vector"
  //stepTo("DOLFIN Vector");



  // Close file
  closeIn();
}
//-----------------------------------------------------------------------------
void DolfinFile::operator<< (const Vector& vector)
{
  openOut();

  


  closeOut();
}
//-----------------------------------------------------------------------------
void DolfinFile::createIndex()
{
  // Do this only once
  if ( objects )
	 return;
  
  char c1 = '\0';
  char c2 = '\0';
  long pos = -1;
  int count = 0;
  std::string word;

  // First time: count number of objects  
  while ( 1 ){
	 
	 // Find next D at the beginning of a line
	 while( !in.eof() ){
		in >> c2;
		if ( c2 == 'D' && ( c1 == '\n' ) || ( c1 == '\0' ) ){
		  pos = in.tellg() - 1;
		  break;
		}
		c1 = c2;
	 }

	 // Reached end of file
	 if ( in.eof() )
		break;
		
	 // Step to beginning of line
	 in.seekg(pos);
	 in >> word;
	 
	 // Should be "DOLFIN"
	 if ( word == "DOLFIN" )
		count++;
	 
  }
  
  // Allocate for list of object positions
  no_objects = count;
  objects = new long[count];

  // Reset file
  in.seekg(ios::beg);
  
  // First time: count number of objects  
  for (int i = 0; i < no_objects; i++){
	 
	 // Find next D at the beginning of a line
	 while( !in.eof() ){
		in >> c2;
		if ( c2 == 'D' && ( c1 == '\n' ) || ( c1 == '\0' ) ){
		  pos = in.tellg() - 1;
		  break;
		}
		c1 = c2;
	 }

	 // Reached end of file
	 if ( in.eof() )
		break;
	 
	 // Step to beginning of line
	 in.seekg(pos);
	 in >> word;
	 
	 // Should be "DOLFIN"
	 if ( word == "DOLFIN" )
		objects[i] = pos;

	 cout << "Found DOLFIN object at position " << pos << endl;
	 
  }

  // Reset file
  in.seekg(ios::beg);  
  
}
//-----------------------------------------------------------------------------
