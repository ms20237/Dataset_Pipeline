from abc import ABC, abstractmethod


class Operator:
    
    def __init__(self, op_name):
        """
        Initialize class parameters.
        Override in subclass if needed.
        """
        super.__init__()
    
    @property
    @abstractmethod
    def op_name(self):
        """
        Return the name of the operator.
        """
        return self.op_name
    
    @property
    @abstractmethod
    def config(ABC):
        raise NotImplementedError("subclass must implement config")
    
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the operation. Must be implemented by subclass.
        """
        pass

    # @abstractmethod
    # def result(self):
    #     """
    #     Return the result of the operation.
    #     Must be implemented by subclass.
    #     """
    #     return self._result


class OperatorType:
    Filter = "Filter"
    Histogram = "Histogram"
    Hist2DVisualizer = "Hist2DVisualizer"
    MultipleFiguresPlotter = "MultipleFiguresPlotter"
    ConfusionMatrixSameSampleVisualizer = "ConfusionMatrixSameSampleVisualizer"
    HistDetectionsSqrtAreaByClass = "HistDetectionsSqrtAreaByClass"
    
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

