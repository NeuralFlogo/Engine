from flogo.data.dataframe import Dataframe


class DeleteOperator:
    def delete(self, dataframe, indexes):
        columns = dataframe.columns
        for index in indexes:
            del columns[index]
        return Dataframe(columns)
