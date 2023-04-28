from flogo.datasets.dataset import Dataset
from pytorch.datasets.transformers.ImageDirectoryNumpyTransformer import ImageDirectoryProcessor
from pytorch.datasets.transformers.NumericNumpyTransformer import NumericNumpyTransformer
from pytorch.preprocesing.ImageProcessor import preprocess_images
from pytorch.preprocesing.ParametersName import *


def __get_array_from(data_string):
    data_list = [x.split(',') for x in data_string.split('\n')]
    data_list.pop()
    return data_list


def __process_images(path, mapper, parameters):
    image_preprocessor = preprocess_images(parameters[IMAGE_SIZE_PARAMETER],
                                           parameters[IMAGE_MEAN_PARAMETER],
                                           parameters[IMAGEN_STD_PARAMETER])
    transformer = ImageDirectoryProcessor(path, image_preprocessor, True)
    return Dataset(transformer.transform_inputs(),
                   transformer.transform_outputs(),
                   mapper,
                   parameters[BATCH_SIZE_PARAMETER])


def read_csv(path):
    with open(path, "r") as archivo:
        return archivo.read()


def __process_numeric(path, mapper, parameters):
    data_array = __get_array_from(read_csv(path))
    transformer = NumericNumpyTransformer(data_array, parameters[PREPROCESSING_PARAMETER],
                                          parameters[SHUFFLE_PARAMETER])
    return Dataset(transformer.transform_inputs(),
                   transformer.transform_outputs(),
                   mapper,
                   parameters[BATCH_SIZE_PARAMETER])


def __process_text(path, mapper, parameters):
    pass


preprocessing_functions = {
    "images": __process_images,
    "numeric": __process_numeric,
    "text": __process_text
}


def read_from_dvc(path, variable_type, mapper, parameters):
    return preprocessing_functions[variable_type](path, mapper, parameters)
