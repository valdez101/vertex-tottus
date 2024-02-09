

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