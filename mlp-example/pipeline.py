

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


@component(packages_to_install=['scikit-learn', 'pandas'], base_image='python:3.9')
def data_preparation(
    dataset: Input[Dataset],
    dataset_transformed: Output[Dataset],
):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder    

    NUMERIC_FEATURE_KEYS = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    LABEL_KEY = 'target'

    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        outputs = inputs.copy()

        # Scale numeric features to range [0, 1]
        scaler = MinMaxScaler()
        outputs[NUMERIC_FEATURE_KEYS] = scaler.fit_transform(outputs[NUMERIC_FEATURE_KEYS])

        # One-hot encode the label column
        encoder = OneHotEncoder()
        outputs[LABEL_KEY] = encoder.fit_transform(outputs[LABEL_KEY].values.reshape(-1, 1)).toarray()

        return outputs

    with open(dataset.path, 'r') as input_file:
        data = pd.read_csv(input_file)

    data_transformed = preprocessing_fn(data)

    with open(dataset_transformed.path, 'w') as f:
        f.write(data_transformed.to_csv(index=False))

@component(packages_to_install=['pandas', 'scikit-learn'],base_image='python:3.9')
def split_data(dataset: Input[Dataset], dataset_train: Output[Dataset], dataset_validation: Output[Dataset], dataset_test: Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    with open(dataset.path, 'r') as input_file:
        data = pd.read_csv(input_file)

    train, test = train_test_split(data, test_size=0.2)
    train, validation = train_test_split(train, test_size=0.2)

    with open(dataset_train.path, 'w') as f:
        f.write(train.to_csv(index=False))
    with open(dataset_validation.path, 'w') as f:
        f.write(validation.to_csv(index=False))
    with open(dataset_test.path, 'w') as f:
        f.write(test.to_csv(index=False))

@component(packages_to_install=['tensorflow', 'pandas', 'joblib'
                                ],base_image='python:3.9')
def train_model(
    dataset_train: Input[Dataset],
    dataset_validation: Input[Dataset],
    model_artifact: Output[Model],
    epochs: int = 10,
):
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    def create_model():
        tf_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(4,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        return tf_model


    with open(dataset_train.path, 'r') as train_file:
        train_data = pd.read_csv(train_file)
    with open(dataset_validation.path, 'r') as validation_file:
        validation_data = pd.read_csv(validation_file)

    # Preprocess the data
    train_features = train_data.drop('target', axis=1)
    train_labels = train_data['target']
    validation_features = validation_data.drop('target', axis=1)
    validation_labels = validation_data['target']

    # Create the model
    tf_model = create_model()

    # Compile the model
    tf_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Train the model
    tf_model.fit(train_features, train_labels, epochs=epochs, validation_data=(validation_features, validation_labels))

    # Evaluate the model
    r = tf_model.evaluate(validation_features, validation_labels)
    print("Result:",r)

    # Save the model
    tf_model.save( "model_artifact.keras")

    # Copy the model to the output path
    import shutil
    shutil.move("model_artifact.keras", model_artifact.path)



        


@component(packages_to_install=['tensorflow', 'pandas'],base_image='python:3.9')
def validate_model(model_artifact: Input[Model], dataset_test: Input[Dataset]):
    import pandas as pd
    import tensorflow as tf

    # Copy the model to the current directory
    import shutil
    shutil.copy(model_artifact.path, "model_artifact.keras")

    # Load the model
    tf_model = tf.keras.models.load_model("model_artifact.keras")

    with open(dataset_test.path, 'r') as test_file:
        test_data = pd.read_csv(test_file)

    # Preprocess the data
    test_features = test_data.drop('target', axis=1)
    test_labels = test_data['target']

    # Evaluate the model
    r = tf_model.evaluate(test_features, test_labels)
    print(r)



@dsl.pipeline(pipeline_root='', name='mlp-example-pipeline')
def ml_pipeline(message: str = 'message'):
    exctract_data_task = exctract_data()
    data_validation_task = data_validation(dataset=exctract_data_task.outputs['dataset'])
    data_preparation_task = data_preparation(dataset=exctract_data_task.outputs['dataset']).after(data_validation_task)
    split_data_task = split_data(dataset=data_preparation_task.outputs['dataset_transformed'])
    train_model_task = train_model(epochs=30, dataset_train=split_data_task.outputs['dataset_train'], dataset_validation=split_data_task.outputs['dataset_validation'])
    validate_model_task = validate_model(model_artifact=train_model_task.outputs['model_artifact'], dataset_test=split_data_task.outputs['dataset_test'])



def run_pipeline():
    host = "http://localhost:9000/pipeline"
    
    # Compile and run the pipeline
    kfp.compiler.Compiler().compile(ml_pipeline, 'mlp-example.yaml')
    kfp.Client(host=host).create_run_from_pipeline_func(ml_pipeline, arguments={})

if __name__ == '__main__':
    run_pipeline()