import factory
import datetime
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


def create_dummy_data(num_rows: int = 10_000) -> pd.DataFrame:
    """Creates a batch of dummy data used for tests and training"""
    return pd.DataFrame(data=DataTypeFakeFactory.create_batch(num_rows))
