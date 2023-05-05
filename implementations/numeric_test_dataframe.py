from flogo.preprocessing.columns.categorical import CategoricalColumn
from flogo.preprocessing.columns.number import NumericColumn
from flogo.preprocessing.map_orchestrator import MapOrchestrator
from flogo.preprocessing.mappers.normalization_mapper import NormalizationMapper
from flogo.preprocessing.mappers.one_hot_mapper import OneHotMapper
from flogo.preprocessing.mappers.standarization_mapper import StandardizationMapper
from flogo.preprocessing.readers.delimeted_file_reader import DelimitedFileReader

path = "C:/Users/Joel/Desktop/kaggle/ds_salaries.csv"
columns = {"work_year": CategoricalColumn(), "experience_level": CategoricalColumn(), "employment_type": CategoricalColumn(),
           "job_title": CategoricalColumn(), "salary": NumericColumn(), "salary_concurrency": CategoricalColumn(),
           "salary_in_usd": NumericColumn(), "employee_residence": CategoricalColumn(), "remote_ratio": NumericColumn(),
           "company_location": CategoricalColumn(), "company_size": CategoricalColumn()}
dataframe = DelimitedFileReader(",").read(path, columns)
dataframe = MapOrchestrator(OneHotMapper(), NormalizationMapper(min=-1, max=1), StandardizationMapper())\
    .process(dataframe, ["work_year", "experience_level", "employment_type", "job_title", "salary_concurrency",
                         "employee_residence", "company_location", "company_size"]
             , ["salary"], ["salary_in_usd", "remote_ratio"])
print(dataframe.get("salary'").values)