%exception {
   try {
      $action
   }
   // all logic_error subclasses
   catch (std::logic_error &e) {
      PyErr_SetString(PyExc_StandardError, const_cast<char*>(e.what()));
      return NULL;
   }
   // all runtime_error subclasses
   catch (std::runtime_error &e) {
      PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
      return NULL;
   }
   // all the rest
   catch (std::exception &e) {
      PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
      return NULL;
   }
}


