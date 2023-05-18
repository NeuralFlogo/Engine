from framework.data.timeline.timeline_builder import TimelineBuilder

STARTER_DELIMITER = "@"


class TimelineReader:
    def __init__(self, parser):
        self.builder = TimelineBuilder()
        self.parser = parser
        self.period = 0

    def read(self, path):
        with open(path) as file:
            for line in file:
                self.pass_line_to_builder(line)
        return self.builder.build()

    def pass_line_to_builder(self, line):
        self.builder.set_command(self.parser.parse(line)) if line.startswith(STARTER_DELIMITER) else \
            self.builder.set(line)
