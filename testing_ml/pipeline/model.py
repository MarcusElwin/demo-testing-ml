from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from typing import Dict

def get_model_pipeline(class_weight: Dict[int, int] = "balanced") -> Pipeline:
    model_steps = [
        (
            "random_forest_clf",
            RandomForestClassifier(
                random_state=123, n_estimators=100, class_weight=class_weight
            ),
        )
    ]
    return Pipeline(model_steps)