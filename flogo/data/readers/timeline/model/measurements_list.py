from flogo.data.readers.timeline.parser import Measurements


class MeasurementsList:
    def __init__(self, measurements: list = []):
        self.measurementsList = [] if len(measurements) == 0 else measurements

    def get_measurements(self):
        return self.measurementsList

    def get(self, index):
        return self.measurementsList[index]

    def __len__(self):
        return len(self.measurementsList)

    def append(self, measurements: Measurements):
        self.measurementsList.append(measurements)
