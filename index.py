BUCKET_NAME ="gs://" + 'vertex-tottus' + "-pipeline"

PATH =%env PATH
%env PATH={PATH}:/home/jupyter/.local/bin
REGION="us-east1"

PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/"

from kfp.dsl import pipeline, component, Artifact, Dataset, Input, Metrics, Model, Output, InputPath, OutputPath
from typing import Optional

@component(
    packages_to_install=["pandas", "pyarrow", "scikit-learn"],
    base_image ="python:3.10",
    output_component_file="dataset_creating_1.yaml"
)
def get_data_from_bq(
    output_data_path: OutputPath("Dataset"),
    bq_table: Optional[str] = None
):

    import pandas as pd
    from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
    iris = load_iris()
    X, Y = iris.data, iris.target

# Crear un DataFrame de Pandas con las características y las etiquetas
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = Y

    df.to_csv(output_data_path, index=False)

@component(
    packages_to_install=["scikit-learn", "pandas", "joblib", "scikit-learn"],
    base_image="python:3.10",
    output_component_file="model_training.yaml",
)
def training_classmod(
    data: Input[Dataset],
    metrics: Output[Metrics],
    model: Output[Model],
    predictions: OutputPath("Dataset")
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from joblib import dump
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    data_encoded=pd.read_csv(data.path)
    
    X_train, X_test, y_train, y_test = train_test_split(data_encoded[iris.feature_names], data_encoded['target'],test_size = 0.3)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred=dt.predict(X_test)
    score=dt.score(X_test, y_test)
    print('accuracy is:', score)
    
    metrics.log_metric("accuracy", (score*100.0))
    metrics.log_metric("model", "Tree Class")
    
    predictions_df = pd.DataFrame({'predicted': y_pred})
    predictions_df.to_csv(predictions, index=False)
    
    dump(dt, model.path + ".joblib")


@pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="custom-pipeline",
)
def pipeline(
    bq_table: str ="",
    output_data_path: str="data.csv",
    project: str = PROJECT_ID,
    region: str = REGION
):
    # Tarea para obtener los datos de BigQuery
    dataset_task = get_data_from_bq(bq_table=bq_table)
    
    # Tarea para entrenar el modelo, calcular predicciones y guardarlas
    training_task = training_classmod(data=dataset_task.output)
        
compiler.Compiler().compile(pipeline_func=pipeline, package_path="custom-pipeline-classifier.json")


run1 = aiplatform.PipelineJob(
    display_name="custom-training-vertex-ai-pipeline",
    template_path="custom-pipeline-classifier.json",
    job_id="custom-pipeline-ef-18",
    enable_caching=False,
)

run1.submit()