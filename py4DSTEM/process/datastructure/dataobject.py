# Defines the DataObject class.
#
# The primary purpose of the DataObject class is to facilitate object level logging.
# Each instance maintains:
#   -a list of parent RawDataCube instances
#   -log indices when the object was created or modified
#   -save info, determining whether the complete data associated with the object is saved
# With respect to save info, note that if the complete data is not saved, the object name and
# log info still is, allowing it to be recreated. Save info must contain separate Boolean values
# for each parent RawDataCube.
#
# All objects containing py4DSTEM data - e.g. RawDataCube, DataCube, DiffractionSlice, 
# RealSlice, and PointList objects - inherit from DataObject.
# Only RawDataCube instances may have an empty parent list.

from ..log import Logger
logger = Logger()

class DataObject(object):
    """
    A DataObject:
        -maintains list of parent RawDataCubes
        -maintins a list of save information for each parent RawDataCube
        -maintains a list of log indices when the object was created/modified
    """
    def __init__(self, parent, default_save_behavior=True):

        self.default_save_behavior = default_save_behavior

        self.parents_and_save_behavior = list()
        self.new_parent(parent=parent, save_behavior=default_save_behavior)

        self.modification_log = list()
        self.log_modification()

    def new_parent(self, parent, save_behavior):
        if parent is not None:
            self.parents_and_save_behavior.append((parent,save_behavior))
            # Add this DataObject to the parent's DataObjectTracker

    def log_modification(self):
        index = self.get_current_log_index()-1
        self.modification_log.append(index)

    @staticmethod
    def get_current_log_index():
        global logger
        return logger.log_index


