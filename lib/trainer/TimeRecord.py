class TimeRecord:
    def __init__(self):
        self.time_record = {}

    def init_record(self, s):
        self.time_record[s] = ([], [])

    def add_record(self, s, start_t, end_t):
        self.time_record[s][0].append(start_t)
        self.time_record[s][1].append(end_t)
