import unittest

from flogo.preprocessing.columns.categorical import CategoricalColumn
from flogo.preprocessing.columns.loaded_image import LoadedImageColumn
from flogo.preprocessing.columns.number import NumericColumn
from flogo.preprocessing.columns.unloaded_image import UnloadedImageColumn
from test.utils import abs_image_path, image_sizes


class ColumnTest(unittest.TestCase):

    def test_categorical_column(self):
        column = CategoricalColumn(["2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023"])
        expected = ["2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023"]
        self.assertEqual(expected, column.get_values())

    def test_number_column(self):
        column = NumericColumn(["85847", "30000", "25500", "175000", "120000",
                                "222200", "136000", "219000", "141000", "147100"])
        expected = [85847, 30000, 25500, 175000, 120000, 222200, 136000, 219000, 141000, 147100]
        self.assertEqual(expected, column.get_values())

    def test_loading_column(self):
        column = UnloadedImageColumn(abs_image_path)
        loaded_column = LoadedImageColumn(column.get_values(), True)
        expected = image_sizes
        self.assertEqual(expected, [image.size for image in loaded_column.get_values()])
