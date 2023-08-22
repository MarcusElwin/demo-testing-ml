import factory
import datetime
import numpy as np
import pandas as pd
from factory import Factory, Faker
from factory.fuzzy import (
    FuzzyChoice,
    FuzzyFloat,
)
from typing import Generic, TypeVar
from dataclasses import dataclass
from enum import Enum
from dateutil.relativedelta import relativedelta

T = TypeVar("T")
FEATURES = 150


class Labels(Enum):
    POSITIVE_CLASS = 1
    NEGATIVE_CLASS = 0

    @staticmethod
    def list():
        return list(map(lambda label: label.value, Labels))


@dataclass
class DataType:
    user_id: str
    date: str
    amount: float
    description: str
    label: int

@dataclass
class DataSet:
    user_id: str
    vector: np.array
    label: int


class BaseFactory(Generic[T], Factory):
    # Base class to enable typing o .create() method
    @classmethod
    def create(cls, **kwargs) -> T:
        return super().create(**kwargs)

class DataTypeFakeFactory(BaseFactory[DataType]):
    class Meta:
        model = DataType

    user_id = Faker("uuid4")
    date = Faker("date_between", start_date="-1y")
    amount = FuzzyFloat(-100, 2000)
    description = Faker("sentence")
    label = FuzzyChoice(Labels.list())


class DataSetFakeFactory(BaseFactory[DataSet]):
    class Meta:
        model = DataSet

    user_id = Faker("uuid4")
    vector = np.random.random((FEATURES, 1))
    label = FuzzyChoice(Labels.list())

class DataSetInvarianceFakeFactory(BaseFactory[DataSet]):
    class Meta:
        model = DataSet

    user_id = Faker("uuid4")
    vector = np.random.uniform(low=0.5, high=1.5, size=(FEATURES, 1))
    label = FuzzyChoice(Labels.list())

def create_dummy_data(num_rows: int = 10_000) -> pd.DataFrame:
    """Creates a batch of dummy data used for tests and training"""
    return pd.DataFrame(data=DataTypeFakeFactory.create_batch(num_rows))

def create_dummy_feature_data(num_rows: int = 10_000, invariance: bool = False) -> pd.DataFrame:
    """Creates a batch of dummy data used for tests and training"""
    if invariance:
        return pd.DataFrame(data=DataSetInvarianceFakeFactory.create_batch(num_rows))
    else:
        return pd.DataFrame(data=DataSetFakeFactory.create_batch(num_rows))

def get_training_data(num_rows: int = 1000, invariance: bool = False) -> np.array:
    df = create_dummy_feature_data(num_rows, invariance)
    features = np.concatenate(df["vector"].values, axis=1)
    target = df["label"].values
    return features, target