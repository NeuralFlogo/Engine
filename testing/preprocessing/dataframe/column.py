import unittest

from flogo.data.columns.categorical import CategoricalColumn
from flogo.data.columns.loaded_image import LoadedImageColumn
from flogo.data.columns.number import NumericColumn
from flogo.data.columns.unloaded_image import UnloadedImageColumn
from testing.utils import image_sizes, abs_image_path


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
