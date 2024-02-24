ONE_HOUR = 60
ONE_DAY = 60 * 24
ONE_MONTH = 60 * 24 * 30
ONE_YEAR = 60 * 24 * 30 * 12
INTERVAL_TIME4LPKT_PLUS = (
    1, 5, 10, 20, 30, 45,
    ONE_HOUR, ONE_HOUR * 2, ONE_HOUR * 4, ONE_HOUR * 8, ONE_HOUR * 12, ONE_HOUR * 16,
    ONE_DAY * 1, ONE_DAY * 2, ONE_DAY * 4, ONE_DAY * 7, ONE_DAY * 10, ONE_DAY * 14, ONE_DAY * 20,
    ONE_MONTH * 1, ONE_MONTH * 2, ONE_MONTH * 3, ONE_MONTH * 6,
    ONE_YEAR * 1, ONE_YEAR * 2
)


ONE_MIN = 60
ONE_HOUR = 60 * 60
USE_TIME4LPKT_PLUS = (
    5, 10, 20, 30, 45,
    ONE_MIN * 1, ONE_MIN * 3, ONE_MIN * 5, ONE_MIN * 10, ONE_MIN * 15, ONE_MIN * 20, ONE_MIN * 30, ONE_MIN * 45,
    ONE_HOUR * 1, ONE_HOUR * 1 + ONE_MIN * 30, ONE_HOUR * 2, ONE_HOUR * 2 + ONE_MIN * 30, ONE_HOUR * 3, ONE_HOUR * 4
)