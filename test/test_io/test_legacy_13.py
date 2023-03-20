from py4DSTEM import read, print_h5_tree, _TESTPATH
from os.path import join


# Set filepaths
filepath = join(_TESTPATH, "v13_sample.h5")




class TestV13:

    # setup/teardown
    def setup_class(cls):
        cls.path = filepath
        pass
    @classmethod
    def teardown_class(cls):
        pass
    def setup_method(self, method):
        pass
    def teardown_method(self, method):
        pass



    def test_print_tree(self):
        print_h5_tree(self.path)


    def test_read_datacube(self):
        d = read(
            self.path,
            datapath='4DSTEM/probe'
        )
        d


    def test_read_braggvectors(self):
        d = read(
            self.path,
            datapath='4DSTEM/braggpeaks_cal'
        )
        d.vectors_uncal
        d.vectors






