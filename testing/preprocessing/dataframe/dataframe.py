import unittest

from flogo.data.columns.categorical import CategoricalColumn
from flogo.data.columns.number import NumericColumn
from flogo.data.dataframe import Dataframe

dataframe = Dataframe({
    "work_year": CategoricalColumn(["2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023"]),
    "experience_level": CategoricalColumn(["SE", "MI", "MI", "SE", "SE", "SE", "SE", "SE", "SE", "SE"]),
    "employment_type": CategoricalColumn(["FT", "CT", "CT", "FT", "FT", "FT", "FT", "FT", "FT", "FT"]),
    "job_title": CategoricalColumn(["Principal_Data_Scientist", "ML_Engineer", "ML_Engineer", "Data_Scientist",
                                    "Data_Scientist", "Applied_Scientist", "Applied_Scientist", "Data_Scientist",
                                    "Data_Scientist", "Data_Scientist"]),
    "salary": NumericColumn([85847, 30000, 25500, 175000, 120000, 222200, 136000, 219000, 141000, 147100])})


class DataframeTest(unittest.TestCase):

    def test_get_method(self):
        self.assertEqual([85847, 30000, 25500, 175000, 120000, 222200, 136000, 219000, 141000, 147100],
                         dataframe.get("salary").get_values())

    def test_len_method(self):
        dataframe.append_column("testing", NumericColumn([85847, 30000, 25500, 175000, 120000,
                                                          222200, 136000, 219000, 141000, 147100]))
        self.assertEqual(5, dataframe.get_column_names().index("testing"))
