// http://uint32t.blogspot.no/2008/03/update-serializing-boosttuple-using.html

#include <boost/tuple/tuple.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/preprocessor/repetition.hpp>

namespace boost { namespace serialization {

#define GENERATE_ELEMENT_SERIALIZE(z,which,unused) \
    ar & boost::serialization::make_nvp("element",t.get< which >());

#define GENERATE_TUPLE_SERIALIZE(z,nargs,unused)                        \
    template< typename Archive, BOOST_PP_ENUM_PARAMS(nargs,typename T) > \
    void serialize(Archive & ar,                                        \
                   boost::tuple< BOOST_PP_ENUM_PARAMS(nargs,T) > & t,   \
                   const unsigned int version)                          \
    {                                                                   \
      BOOST_PP_REPEAT_FROM_TO(0,nargs,GENERATE_ELEMENT_SERIALIZE,~)     \
    }

    BOOST_PP_REPEAT_FROM_TO(1,10,GENERATE_TUPLE_SERIALIZE,~)

}}
