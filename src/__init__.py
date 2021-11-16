from .main import *
from .mainModel import *
from .augmentData import *
from .getSIDMdata import *
from .globalVariables import *
from .inceptionModules import *
from .tools import *
import pkg_resources  # part of setuptools                                                                                        

__version__ = pkg_resources.require("darkCNN")[0].version
