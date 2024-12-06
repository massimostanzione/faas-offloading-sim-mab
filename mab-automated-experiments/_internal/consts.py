from enum import Enum

PREFIX_STATSFILE = "stats"
PREFIX_MABSTATSFILE = "mab-stats"
SUFFIX_TXT = ".txt"
SUFFIX_JSON = ".json"
SUFFIX_STATSFILE = SUFFIX_TXT
SUFFIX_MABSTATSFILE = SUFFIX_JSON

DELIMITER_COMMA = ', '
DELIMITER_HYPHEN = '-'
DELIMITER_AXIS = '>'
DELIMITER_PARAMS = '='

class ExecMode(Enum):
    NONE = 'none'
    AUTOMATED = 'automated'

class RundupBehavior(Enum):
    NO = "no"
    ALWAYS = "always"
    SKIP_EXISTENT = "skip-existent"

class RewardFnAxis(Enum):
    LOADIMB = "load_imb"
    RESPONSETIME = "rt"
    COST = "cost"
    UTILITY = "utility"
    VIOLATIONS = "load_imb"


PIPELINE_FILE = "pipeline.txt"
EXPCONF_FILE = "expconf.ini"
CONFIG_FILE = "config.ini"
