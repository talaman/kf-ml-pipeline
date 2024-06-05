

from typing import Dict, List

from kfp import dsl
from kfp.dsl import Input, InputPath, Output, OutputPath, Dataset, Model, component, HTML
import kfp


@component(packages_to_install=['scikit-learn','pandas'])
def exctract_data(
    dataset: Output[Dataset],
):
    from sklearn import datasets
    data = datasets.load_iris(as_frame=True)
    data = data.frame
    with open(dataset.path, 'w') as f:
        f.write(data.to_csv())

@component(packages_to_install=['evidently'],base_image='python:3.9')
def data_validation(
    dataset: Input[Dataset],
    tests: Output[HTML],
    reports: Output[HTML],
):
    from evidently import ColumnMapping
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataQualityTestPreset
    from evidently.test_preset import DataStabilityTestPreset
    from evidently.report import Report
    from evidently.metric_preset import DataQualityPreset
    import pandas as pd
    
    with open(dataset.path, 'r') as input_file:
        data = pd.read_csv(input_file)

    tests_suite= TestSuite(tests=[
        DataStabilityTestPreset(),
        DataQualityTestPreset()
    ])
    tests_suite.run(current_data=data.iloc[:60], reference_data=data.iloc[60:], column_mapping=None)
    tests_suite.save_html(tests.path)


    reports_suite = Report(metrics=[
        DataQualityPreset(),
    ])

    reports_suite.run(current_data=data.iloc[:60], reference_data=data.iloc[60:], column_mapping=None)
    reports_suite.save_html(reports.path)


@dsl.pipeline(pipeline_root='', name='mlp-example-pipeline')
def data_passing_pipeline(message: str = 'message'):
    exctract_data_task = exctract_data()
    data_validation_task = data_validation(dataset=exctract_data_task.outputs['dataset'])

def run_pipeline():
    host = "http://localhost:9000/pipeline"
    
    # Compile and run the pipeline
    kfp.compiler.Compiler().compile(data_passing_pipeline, 'mlp-example.yaml')
    kfp.Client(host=host).create_run_from_pipeline_func(data_passing_pipeline, arguments={})

if __name__ == '__main__':
    run_pipeline()