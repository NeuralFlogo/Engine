from datetime import datetime


class Instant:
    def __init__(self, value):
        self.value = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%fZ").timestamp()

    def get_value(self):
        return self.value
