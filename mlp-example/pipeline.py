

from typing import Dict, List

from kfp import dsl
from kfp.dsl import Input, InputPath, Output, OutputPath, Dataset, Model, component, HTML
import kfp


@component(packages_to_install=['scikit-learn','pandas'])
def exctract_data(
    dataset: Output[Dataset],
):
    from sklearn import datasets
    data = datasets.load_iris(as_frame=True).frame
    with open(dataset.path, 'w') as f:
        f.write(data.to_csv(index=False))

@component(packages_to_install=['evidently'],base_image='python:3.10')
def data_validation(
    dataset: Input[Dataset],
    tests: Output[HTML],
    reports: Output[HTML],
):
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataQualityTestPreset
    from evidently.test_preset import DataStabilityTestPreset
    from evidently.report import Report
    from evidently.metric_preset import DataQualityPreset
    from evidently.metric_preset import DataDriftPreset
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
        DataDriftPreset()
    ])

    reports_suite.run(current_data=data.iloc[:60], reference_data=data.iloc[60:], column_mapping=None)
    reports_suite.save_html(reports.path)


@component(packages_to_install=['tensorflow_transform', 'pandas'],base_image='python:3.9')
def data_preparation(
    dataset: Input[Dataset],
    dataset_transformed: Output[Dataset],
):
    import tensorflow_transform as tft
    import tensorflow as tf
    import pandas as pd
    tf.compat.v1.disable_eager_execution()

    NUMERIC_FEATURE_KEYS = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    LABEL_KEY = 'target'
    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        outputs = inputs.copy()
        # for key in NUMERIC_FEATURE_KEYS:
        #     # outputs[key] = tft.scale_to_0_1(outputs[key])
        #     outputs[key] = tft.scale_to_z_score(tf.cast(outputs[key], tf.float32))


        # outputs[LABEL_KEY] = tf.one_hot(outputs[LABEL_KEY], depth=3)  # Assuming 3 classes                                 

        return outputs   
    
            
    with open(dataset.path, 'r') as input_file:
        data = pd.read_csv(input_file)

    
    data_transformed = preprocessing_fn(data)

    with open(dataset_transformed.path, 'w') as f:
        f.write(data_transformed.to_csv(index=False))




@dsl.pipeline(pipeline_root='', name='mlp-example-pipeline')
def data_passing_pipeline(message: str = 'message'):
    exctract_data_task = exctract_data()
    data_validation_task = data_validation(dataset=exctract_data_task.outputs['dataset'])
    data_preparation_task = data_preparation(dataset=exctract_data_task.outputs['dataset']).after(data_validation_task)

def run_pipeline():
    host = "http://localhost:9000/pipeline"
    
    # Compile and run the pipeline
    kfp.compiler.Compiler().compile(data_passing_pipeline, 'mlp-example.yaml')
    kfp.Client(host=host).create_run_from_pipeline_func(data_passing_pipeline, arguments={})

if __name__ == '__main__':
    run_pipeline()