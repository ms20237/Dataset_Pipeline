from typing import List
import plotly.graph_objs as go

import fiftyone as fo
from fiftyone import ViewField as F

from shuttrix.operators.base import Operator


class Filter(Operator):
    """
        Operator to create a view containing samples of specified splits and classes.

        This operator filters a dataset view to include only samples that belong to the
        specified splits and classes. It can be used to create a subset of the dataset
        for analysis or visualization purposes.

        Args:
            op_name (str): The name of the operator.
            splits (list[str], optional): The list of splits to include. Defaults to None.
            classes (list[str], optional): The list of classes to include. Defaults to None.

        Attributes:
            op_name (str): The name of the operator.
            op_type (OperatorType): The type of the operator.
            config (dict): The configuration of the operator.
            result (fo.DatasetView): The resulting filtered dataset view.

        Methods:
            execute(view: fo.DatasetView): Executes the operator on the given dataset view.
    """

    def __init__(
        self, op_name: str, splits: list[str] = None, classes: list[str] = None
    ) -> None:
        """
        Initialize the Filter operator.

        Args:
            op_name (str): The name of the operator.
            splits (list[str], optional): The list of splits. Defaults to None.
            classes (list[str], optional): The list of classes. Defaults to None.
        """
        # self._op_type = OperatorType.DeletedFFPViewCreator
        self._op_name = op_name
        self._config = {
            "splits": splits,
            "classes": classes,
        }
        self._result = None

    @property
    def op_name(self) -> str:
        """
        Get the name of the operator.

        Returns:
            str: The name of the operator.
        """
        return self._op_name

    @property
    def config(
        self,
    ) -> dict:
        """
        Get the configuration of the operator.

        Returns:
            dict: The configuration of the operator.
        """
        return self._config

    def execute(self, view: fo.DatasetView):
        """
        Execute the operator on the given dataset view.

        Args:
            view (fo.DatasetView): The dataset view to operate on.
        """
        # Classes are defined in list
        if self._config["classes"] and self.config["splits"]:
            self._result = (
                view.match_tags(self.config["splits"])
                .match(F("ground_truth.detections").length() > 0)
                .match(
                    F("ground_truth.detections.label").contains(self._config["classes"])
                )
            )

        # No splits so assume all splits
        elif self._config["classes"]:
            self._result = view.match(F("ground_truth.detections").length() > 0).match(
                F("ground_truth.detections.label").contains(self._config["classes"])
            )

        # No classes so assume all classes
        elif self._config["splits"]:
            self._result = view.match_tags(self.config["splits"])

        else:
            self._result = view

    @property
    def result(self):
        """
        Get the result of the operator.

        Returns:
            The result of the operator.
        """
        return self._result

