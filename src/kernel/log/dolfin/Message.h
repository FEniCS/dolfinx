// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>

namespace dolfin
{

  /// A message is a string message which is displayed
  /// only a limited number of times.
  ///
  /// Example of usage:
  ///
  ///   Message message("System is stiff, damping is needed.");
  ///   while ()
  ///   {
  ///     ...
  ///     if ( ... )
  ///     {
  ///       message.display();
  ///       ...
  ///     }
  ///   }

  class Message
  {
  public:

    /// Constructor
    Message(const std::string message, unsigned int maxcount = 1);

    /// Destructor
    ~Message();

    /// Display message
    void display();
    
    /// Display count
    unsigned int count() const;

    /// Maximum display count
    unsigned int maxcount() const;    

  private:

    std::string message;
    unsigned int _maxcount;
    unsigned int _count;

  };

}
