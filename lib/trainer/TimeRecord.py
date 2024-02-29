import time


class TimeRecord:
    def __init__(self):
        self.time_record = {}
        self.time_record_name = []

    def add_record(self, name):
        if name in self.time_record_name:
            self.time_record[name].append(time.time_ns())
        else:
            self.time_record[name] = [time.time_ns()]
            self.time_record_name.append(name)

    def parse_time(self):
        result = [0] * (len(self.time_record_name) - 1)
        for i in range(len(self.time_record[self.time_record_name[0]])):
            for j in range(len(self.time_record_name) - 1):
                last_time = self.time_record[self.time_record_name[j]][i]
                next_time = self.time_record[self.time_record_name[j + 1]][i]
                result[j] += (next_time - last_time)

        for k in range(len(self.time_record_name) - 1):
            last_name = self.time_record_name[k]
            next_name = self.time_record_name[k + 1]
            print(f"{last_name} to {next_name}: {result[k] / 1000000000} s, ")

        self.time_record = {}
        self.time_record_name = []

