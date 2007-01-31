try:
    from viper.viper_dolfin import *
except:
    def plot(*args):
        print "Please install viper: http://www.fenics.org/dev/viper/"

    class Plotter:
        def __getattr__(self, name):

            def method(*args, **kwargs):
                print "Please install viper: http://www.fenics.org/dev/viper/"

            return method
