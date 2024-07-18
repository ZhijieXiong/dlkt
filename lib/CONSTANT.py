ONE_HOUR = 60
ONE_DAY = 60 * 24
ONE_MONTH = 60 * 24 * 30
ONE_YEAR = 60 * 24 * 30 * 12

FORGET_POINT = (
    5, 10, 20, ONE_HOUR, 9 * ONE_HOUR, ONE_DAY, 2 * ONE_DAY, 6 * ONE_DAY, ONE_MONTH
)
REMAIN_PERCENT = (
    0.9, 0.79, 0.58, 0.44, 0.36, 0.33, 0.28, 0.25, 0.21
)

INTERVAL_TIME4LPKT_PLUS = (
    5, 10, 20, 40,
    ONE_HOUR, ONE_HOUR * 2, ONE_HOUR * 4, ONE_HOUR * 8, ONE_HOUR * 12, ONE_HOUR * 16,
    ONE_DAY * 1, ONE_DAY * 2, ONE_DAY * 4, ONE_DAY * 7, ONE_DAY * 10, ONE_DAY * 14, ONE_DAY * 20,
    ONE_MONTH * 1, ONE_MONTH * 2, ONE_MONTH * 3, ONE_MONTH * 6,
    ONE_YEAR * 1
)


ONE_MIN = 60
ONE_HOUR = 60 * 60
USE_TIME4LPKT_PLUS = (
    5, 10, 20, 30, 45,
    ONE_MIN * 1, ONE_MIN * 3, ONE_MIN * 5, ONE_MIN * 10, ONE_MIN * 15, ONE_MIN * 20, ONE_MIN * 30, ONE_MIN * 45,
    ONE_HOUR * 1, ONE_HOUR * 1 + ONE_MIN * 30, ONE_HOUR * 2, ONE_HOUR * 2 + ONE_MIN * 30, ONE_HOUR * 3
)

HAS_TIME = ["assist2012", "assist2017", "junyi2015", "slepemapy", "ednet-kt1", "edi2020-task1", "edi2020-task34",
            "algebra2005", "algebra2006", "algebra2008", "bridge2algebra2006", "bridge2algebra2008", "xes3g5m",
            "statics2011"]
HAS_USE_TIME = ["assist2012", "assist2009", "assist2017", "junyi2015", "slepemapy", "ednet-kt1", "algebra2005",
                "bridge2algebra2006", "SLP-chi", "SLP-his", "SLP-mat", "SLP-geo", "SLP-eng", "SLP-phy", "SLP-bio"]
HAS_NUM_HINT = ["assist2012", "assist2009", "assist2017", "junyi2015", "algebra2005", "bridge2algebra2006", "statics2011"]
HAS_NUM_ATTEMPT = ["assist2012", "assist2009", "assist2017", "junyi2015"]
HAS_AGE = ["edi2020-task1", "edi2020-task34"]
HAS_CORRECT_FLOAT = ["SLP-chi", "SLP-his", "SLP-mat", "SLP-geo", "SLP-eng", "SLP-phy", "SLP-bio"]
HAS_QUESTION_MODE = ["junyi2015", "SLP-chi", "SLP-his", "SLP-mat", "SLP-geo", "SLP-eng", "SLP-phy", "SLP-bio"]

MODEL_USE_QC = ["qDKT", "SAINT", "AKT", "LPKT", "DIMKT", "SimpleKT", "AT_DKT", "QIKT", "LBKT", "DCT", "AuxInfoDCT",
                "qDKT_CORE", "ELMKT", "AuxInfoQDKT", "LBMKT", "IDCT"]
