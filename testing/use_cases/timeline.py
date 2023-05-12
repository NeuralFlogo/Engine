from flogo.data.readers.timeline.parser import Parser
from flogo.data.readers.timeline.timeline_reader import TimelineReader
from flogo.data.readers.timeline.utils.units import *

timeline = TimelineReader(Parser()).read("/Users/jose_juan/Downloads/kraken.its")

dataframe = timeline.group_by(1, YEAR).to_dataframe(5)

print(dataframe.column_names())
