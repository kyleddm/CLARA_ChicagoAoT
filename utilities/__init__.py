__all__=['calcSecDiff','pruneTime','parse_json_args','extract_metadata','extractSensorType','__version__','author','magic_number','DEFAULT_CONFIG','timestart_str','config']
import json 
from .utilities import calcSecDiff
from .utilities import pruneTime
from .utilities import parse_json_args
from .utilities import extract_metadata
from .utilities import extractSensorType
from .utilities import DEFAULT_CONFIG
from .utilities import timestart_str
from .utilities import config
#from . import utilities
__version__ = '1.0.0'
author= 'DeMedeiros, Kyle D'
magic_number = 42
#DEFAULT_CONFIG='./config.json'
#timestart_str = '2025/01/01 00:00:00'
#config={}#This must be initialized when the package is loaded
print(f"{__name__} Initializing package version {__version__}")

