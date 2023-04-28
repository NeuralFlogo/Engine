import os

from dvc import api

PUBLIC_KEY = 'AWS_ACCESS_KEY_ID'
PRIVATE_KEY = 'AWS_SECRET_ACCESS_KEY'


class DvcReader:

    def __init__(self, file_path, keys_path, repo):
        self.file_path = file_path
        self.keys_path = keys_path
        self.repo = repo
        self.__set_keys()

    def __read_keys(self):
        with open(self.keys_path, "r") as f:
            lines = f.read().split("\n")
            return lines[1].split(" ")[2], lines[2].split(" ")[2]

    def __set_keys(self):
        public_key, private_key = self.__read_keys()
        os.environ[PUBLIC_KEY] = public_key
        os.environ[PRIVATE_KEY] = private_key

    def read(self):
        with api.open(self.file_path, repo=self.repo) as f:
            return self.__get_array_from(f.read())

    def download(self):
        pass

    @staticmethod
    def __get_array_from(data_string):
        data_list = [x.split(',') for x in data_string.split('\n')]
        data_list.pop()
        return data_list

    # def download_directory(self, directory_path):
    #     pass
