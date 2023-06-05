import os


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


image_path = ["/resources/image_data/cat/cat.6.jpg", "/resources/image_data/cat/cat.7.jpg",
              "/resources/image_data/cat/cat.5.jpg", "/resources/image_data/cat/cat.4.jpg",
              "/resources/image_data/cat/cat.1.jpg", "/resources/image_data/cat/cat.3.jpg",
              "/resources/image_data/cat/cat.2.jpg", "/resources/image_data/cat/cat.10.jpg",
              "/resources/image_data/cat/cat.9.jpg", "/resources/image_data/cat/cat.8.jpg",
              "/resources/image_data/dog/dog.1.jpg", "/resources/image_data/dog/dog.2.jpg",
              "/resources/image_data/dog/dog.3.jpg", "/resources/image_data/dog/dog.4.jpg",
              "/resources/image_data/dog/dog.5.jpg", "/resources/image_data/dog/dog.6.jpg",
              "/resources/image_data/dog/dog.7.jpg", "/resources/image_data/dog/dog.8.jpg",
              "/resources/image_data/dog/dog.9.jpg", "/resources/image_data/dog/dog.10.jpg"]

image_sizes = [(400, 303), (495, 499), (175, 144), (499, 375), (300, 280), (500, 414), (312, 396), (489, 499), (320, 425),
               (461, 345), (300, 287), (499, 376), (269, 292), (135, 101), (500, 380), (272, 335), (499, 371), (499, 403),
               (274, 500), (499, 375)]

abs_image_path = [abs_path(path) for path in image_path]
