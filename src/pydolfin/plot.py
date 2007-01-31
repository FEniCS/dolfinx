try:
    from viper import viper_dolfin as viper
except:
    class ViperDummy(object):
        def __getattr__(self, name):
            def method(*args, **kwargs):
                print "Please install viper: http://www.fenics.org/viper"
                pass
            return method


    viper = ViperDummy()


