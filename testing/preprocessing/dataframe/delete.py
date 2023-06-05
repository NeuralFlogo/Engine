import unittest

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.dataframe import Dataframe
from framework.preprocessing.delete_column import DeleteOperator

columns = {"work_year": CategoricalColumn([2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]),
           "experience_level": CategoricalColumn(["SE", "MI", "MI", "SE", "SE", "SE", "SE", "SE", "SE", "SE"])}


class DeleteTest(unittest.TestCase):

    def test(self):
        dataframe = Dataframe(columns)
        DeleteOperator().delete(dataframe, ["work_year"])
        self.assertEqual(["experience_level"], dataframe.column_names())
