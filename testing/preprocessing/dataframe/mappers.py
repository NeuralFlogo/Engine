import unittest

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.columns.unloaded_image import UnloadedImageColumn
from framework.data.dataframe.dataframe import Dataframe
from framework.preprocessing.mappers.composite import CompositeMapper
from framework.preprocessing.mappers.leaf.grayscale_mapper import GrayScaleMapper
from framework.preprocessing.mappers.leaf.normalization_mapper import NormalizationMapper
from framework.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from framework.preprocessing.mappers.leaf.resize_mapper import ResizeMapper
from framework.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from framework.preprocessing.mappers.leaf.type_mapper import TypeMapper
from pytorch.preprocessing.mappers.tensor_mapper import TensorMapper
from testing.utils import abs_image_path, image_sizes

csv_dataframe = Dataframe({
    "work_year": CategoricalColumn(["2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2023", "2022"]),
    "experience_level": CategoricalColumn(["SE", "MI", "MI", "SE", "SE", "SE", "SE", "SE", "SE", "SE"]),
    "employment_type": CategoricalColumn(["FT", "CT", "CT", "FT", "FT", "FT", "FT", "FT", "FT", "FT"]),
    "job_title": CategoricalColumn(["Principal_Data_Scientist", "ML_Engineer", "ML_Engineer", "Data_Scientist",
                                    "Data_Scientist", "Applied_Scientist", "Applied_Scientist", "Data_Scientist",
                                    "Data_Scientist", "Data_Scientist"]),
    "salary": NumericColumn([85847, 30000, 25500, 175000, 120000, 222200, 136000, 219000, 141000, 147100])})

image_dataframe = Dataframe({"input": UnloadedImageColumn(abs_image_path),
                             "output": CategoricalColumn(
                                 ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat',
                                  'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog']),
                             "loaded_input": LoadedImageColumn(abs_image_path, True)})


class MappersTest(unittest.TestCase):

    def test_one_hot_mapper(self):
        onehot_mapper = OneHotMapper()
        dataframe0 = onehot_mapper.map(csv_dataframe, ["work_year"])
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.assertEqual(expected, dataframe0.get("work_year_2022").get_values())

    def test_normalization_mapper(self):
        normalization_mapper = NormalizationMapper()
        dataframe1 = normalization_mapper.map(csv_dataframe, ["salary"])
        expected = [0.306797153024911, 0.022877478393492627, 0.0, 0.7600406710726996,
                    0.4804270462633452, 1.0, 0.5617691916624301, 0.983731570920183,
                    0.5871886120996441, 0.6182003050330452]
        self.assertEqual(expected, dataframe1.get("salary'").get_values())

    def test_standardization_mapper(self):
        standardization_mapper = StandardizationMapper()
        dataframe2 = standardization_mapper.map(csv_dataframe, ["salary"])
        expected = [-0.6849327792423494, -1.5480515990896675, -1.6175993758603613, 0.6929323190771297,
                    -0.15709606367579337, 1.4224112220941838, 0.09018492039778427, 1.3729550252794682,
                    0.16746022792077728, 0.2617361030988287]
        self.assertEqual(expected, dataframe2.get("salary'").get_values())

    def test_type_mapper(self):
        type_mapper = TypeMapper(LoadedImageColumn)
        dataframe3 = type_mapper.map(image_dataframe, ["input"])
        expected = image_sizes
        self.assertEqual(expected, [image.size for image in dataframe3.get("input'").get_values()])

    def test_grayscale_mapper(self):
        gray_scale_mapper = GrayScaleMapper()
        dataframe4 = (gray_scale_mapper.map(image_dataframe, ["loaded_input"]))
        expected = ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
        self.assertEqual(expected, [image.mode for image in dataframe4.get("loaded_input'").get_values()])

    def test_resize_mapper(self):
        resize_mapper = ResizeMapper((50, 50))
        dataframe5 = resize_mapper.map(image_dataframe, ["loaded_input"])
        expected = [(50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50),
                    (50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50),
                    (50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50)]
        self.assertEqual(expected, [image.size for image in dataframe5.get("loaded_input'").get_values()])

    def test_to_tensor_mapper(self):
        pytorch_tensor_mapper = TensorMapper()
        dataframe6 = pytorch_tensor_mapper.map(image_dataframe, ["loaded_input"])
        expected = [[3, 303, 400], [3, 499, 495], [3, 144, 175], [3, 375, 499], [3, 280, 300], [3, 414, 500],
                    [3, 396, 312], [3, 499, 489], [3, 425, 320], [3, 345, 461], [3, 287, 300], [3, 376, 499],
                    [3, 292, 269], [3, 101, 135], [3, 380, 500], [3, 335, 272], [3, 371, 499], [3, 403, 499],
                    [3, 500, 274], [3, 375, 499]]
        self.assertEqual(expected, [list(tensor.shape) for tensor in dataframe6.get("loaded_input'").get_values()])

    def test_composite_mapper(self):
        composite_mapper = CompositeMapper([(ResizeMapper((70, 50))), GrayScaleMapper()])
        dataframe7 = composite_mapper.map(image_dataframe, ["loaded_input"])
        expected = [[(70, 50), (70, 50), (70, 50), (70, 50), (70, 50), (70, 50), (70, 50),
                     (70, 50), (70, 50), (70, 50), (70, 50), (70, 50), (70, 50), (70, 50),
                     (70, 50), (70, 50), (70, 50), (70, 50), (70, 50), (70, 50)],
                    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']]
        self.assertEqual(expected, [[image.size for image in dataframe7.get("loaded_input'").get_values()],
                                    [image.mode for image in dataframe7.get("loaded_input'").get_values()]])
