import os
import time
import itertools
import webbrowser
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

from collections import defaultdict

import fiftyone as fo
from fiftyone import ViewField as F

from shuttrix.operators.base import Operator, OperatorType


class Hist2DVisualizer(Operator):
    """
        Hist2DVisualizer is an Operator for visualizing a 2D histogram of two specified fields in a dataset.
        This class generates a 2D histogram plot using Plotly, allowing for customization of the number of bins,
        axis ranges, labels, title, and legend. It can optionally display the plot interactively.
            op_name (str): The name of the operator.
            show (bool): Whether to display the plot interactively.
            nxbins (int): Number of bins along the x-axis.
            nybins (int): Number of bins along the y-axis.
            xrange (float): The range of values to include on the x-axis.
            yrange (float): The range of values to include on the y-axis.
            xaxis (str): The label for the x-axis.
            yaxis (str): The label for the y-axis.
            title (str): The title of the plot.
            legend (str): The legend title for the plot.
       
        Attributes:
            op_name (str): The name of the operator.
            op_type (OperatorType): The type of the operator.
            config (dict): The configuration dictionary for the operator.
            result (go.Figure): The resulting Plotly figure after execution.
       
        Methods:
            execute(xdata: List[float], ydata: List[float]) -> None:
                Generates and optionally displays a 2D histogram plot from the provided x and y data.
    
    """

    def __init__(
        self,
        op_name: str,
        show: bool,
        nxbins: int,
        nybins: int,
        xrange: float,
        yrange: float,
        xaxis: str,
        yaxis: str,
        title: str,
        legend: str,
        split: str = None,
        step: str = "General",
        use_hist2d: bool = True 
    ) -> None:
        self._op_name = op_name
        self._op_type = OperatorType.Hist2DVisualizer
        self._config = {
            "xrange": xrange,
            "xaxis": xaxis,
            "yrange": yrange,
            "yaxis": yaxis,
            "show": show,
            "nxbins": nxbins,
            "nybins": nybins,
            "title": title,
            "legend": legend,
            "split": split,
            "step": step,
            "use_hist2d": use_hist2d,
        }
        self._result = None

    @property
    def op_name(self):
        """str: The name of the operator."""
        return self._op_name

    @property
    def op_type(self):
        """OperatorType: The type of the operator."""
        return self._op_type

    @property
    def config(
        self,
    ):
        """dict: The configuration of the operator."""
        return self._config

    def _get_gt_eq_from_det(self, det, view: fo.DatasetView, sample_id, eval_key):
        """_summary_

        i couldnt find way to get detection directly by id so pass sample_id to not search whole samples for a detection

        Args:
            det (_type_): _description_
            view (fo.DatasetView): _description_
            sample_id (_type_): _description_
            eval_key (_type_): _description_

        Returns:
            _type_: _description_
        """
        gt_eq_id = det[f"{eval_key}_id"]
        sample = view[sample_id]
        for gt in sample["ground_truth"]["detections"]:
            if gt["id"] == gt_eq_id:
                return gt

    def execute(self, xdata: List[float], ydata: List[float]) -> None:
        """Plot 2D histogram from xdata and ydata and optionally show/save it."""
        layout = go.Layout(
            title=self._config["title"],
            xaxis=dict(title=self._config["xaxis"]),
            yaxis=dict(title=self._config["yaxis"]),
            legend=dict(title=self._config["legend"]),
        )
        
        if self._config["use_hist2d"]:
            fig = go.Figure(data=[go.Histogram2d(
                x=xdata,
                y=ydata,
                xbins=dict(
                    start=0, end=self._config["xrange"],
                    size=self._config["xrange"] / self._config["nxbins"]
                ),
                ybins=dict(
                    start=0, end=self._config["yrange"],
                    size=self._config["yrange"] / self._config["nybins"]
                ),
                colorscale='Blues',
            )], layout=layout)

        else:
            fig = go.Figure(data=[go.Bar(
                x=xdata,
                y=ydata,
                marker_color='skyblue'
            )], layout=layout)
            
        fig.update_layout(title=self._config["title"])
        self._result = fig

        os.makedirs("./plots", exist_ok=True)
        
        split = self._config["split"]
        step = self._config["step"]
        if self._config["split"]:
            if self._config["show"]:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                html_path = f"./plots/{self._op_name}_{split}_{timestamp}_{step}_plot.html"
                fig.write_html(html_path)
                print(f"[INFO] Plot saved to {html_path}")
                file_uri = Path(html_path).resolve().as_uri()
                webbrowser.open(file_uri)
        else:
            if self._config["show"]:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                html_path = f"./plots/{self._op_name}_{timestamp}_{step}_plot.html"
                fig.write_html(html_path)
                print(f"[INFO] Plot saved to {html_path}")
                file_uri = Path(html_path).resolve().as_uri()
                webbrowser.open(file_uri)
            
    @property
    def result(self):
        """go.Figure: The result of the operator."""
        return self._result


class ConfusionMatrixSameSampleVisualizer(Operator):
    """
        ConfusionMatrixSameSampleVisualizer is an Operator for visualizing the confusion matrix of predictions
        against ground truth labels for a specific class in a FiftyOne dataset.
        This operator filters a FiftyOne dataset view to select samples with a specific class,
        extracts the predicted and ground truth labels, and generates a confusion matrix using Plotly.
            op_name (str): The name of the operator instance.
            pred_key (str): The key in the sample dict for predicted labels.
            gt_key (str): The key in the sample dict for ground truth labels.
            class_name (str): The class name to filter and visualize.
       
        Attributes:
            _op_name (str): The operator's name.
            _op_type (OperatorType): The type of the operator.
            _config (dict): Configuration dictionary containing pred_key, gt_key, and class_name.
            _result: The generated confusion matrix figure.
       
        Properties:
            op_name (str): Returns the operator's name.
            op_type (OperatorType): Returns the operator's type.
            config (dict): Returns the operator's configuration.
            result: Returns the generated plot.
       
        Methods:
            execute(view: fo.DatasetView): Executes the operator on the given dataset view, filters for the specified class,
                extracts relevant data, and generates the confusion matrix visualization.
        
    """
    def __init__(
        self,
        op_name: str,
        show: bool,
        title: str,
        legend: str,
        split: str = None,
        step: str = "General"
    ) -> None:
        self._op_name = op_name
        self._op_type = OperatorType.ConfusionMatrixSameSampleVisualizer
        self._config = {
            "show": show,
            "title": title,
            "legend": legend,
            "split": split,
            "step": step
        }
        self._result = None

    @property
    def op_name(self) -> str:
        """str: The name of the operator."""
        return self._op_name

    @property
    def op_type(self) -> OperatorType:
        """OperatorType: The type of the operator."""
        return self._op_type

    @property
    def config(
        self,
    ) -> dict:
        """dict: The configuration of the operator."""
        return self._config

    def execute(self, view: fo.DatasetView):
        """Generate and display a confusion matrix for the specified class."""
        step = self._config["step"]
        dataset = view
        
        # Filter by split if specified
        if self._config["split"]:
            dataset = dataset.match_tags(self._config["split"])
        
        # Extract predictions and ground truth labels
        label_counts = defaultdict(int)
        co_occurrence = defaultdict(int)
        
        # Extract label sets per sample
        for sample in dataset:
            labels = set()
            
            # Example for detection task
            if hasattr(sample.ground_truth, "detections"):
                labels = set(d.label for d in sample.ground_truth.detections)
            
            # Count co-occurrence
            for label in labels:
                label_counts[label] += 1
            for l1, l2 in itertools.combinations(sorted(labels), 2):
                co_occurrence[(l1, l2)] += 1
                co_occurrence[(l2, l1)] += 1  # symmetric
        
        # Handle empty data case
        if len(label_counts) == 0:
            print(f"[WARNING] No labels found in split '{self._config['split']}'. Skipping co-occurrence matrix.")
            return
    
        # Create matrix
        all_labels = sorted(label_counts.keys())
        label_idx = {l: i for i, l in enumerate(all_labels)}
        n = len(all_labels)
        matrix = np.zeros((n, n))

        for (l1, l2), count in co_occurrence.items():
            i, j = label_idx[l1], label_idx[l2]
            matrix[i][j] = count
        
        # Plot with Plotly
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_labels,
            y=all_labels,
            colorscale=[[0, 'white'], [1, 'blue']],
            text=matrix.astype(int),
            texttemplate="%{text}",
            zmin=0,
            zmax=matrix.max(),
            colorbar=dict(title=self._config["title"])
        ))

        fig.update_layout(
            title=self._config["title"],
            xaxis_title="Label",
            yaxis_title="Label",
        )
        
        if self._config["show"]:
            if self._config["split"]:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                html_path = f"./plots/{self._op_name}_{self._config['split']}_{timestamp}_{step}_plot.html"
                fig.write_html(html_path)
                print(f"[INFO] Plot saved to {html_path}")
                file_uri = Path(html_path).resolve().as_uri()
                webbrowser.open(file_uri)
            else:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                html_path = f"./plots/{self._op_name}_{timestamp}_{step}_plot.html"
                fig.write_html(html_path)
                print(f"[INFO] Plot saved to {html_path}")
                file_uri = Path(html_path).resolve().as_uri()
                webbrowser.open(file_uri)
        
        self._result = fig
    @property
    def result(self):
        """go.Figure: The result of the operator."""
        return self._result    
        

class ConfElDistTPVisualizer(Operator):
    """
        ConfElDistTPVisualizer is an Operator for visualizing the relationship between detection confidence,
        a ground truth element attribute, and distance to target for true positive detections.
        This operator filters a FiftyOne dataset view to select true positive detections of a specific class,
        extracts triplets of (confidence, element attribute, distance to target) for each detection, and
        generates a 3D scatter plot using Plotly.
            op_name (str): The name of the operator instance.
            pred_key (str): The key in the sample dict for predicted detections.
            eval_key (str): The key used for evaluation (e.g., to determine true positives).
            el_key (str): The key in the ground truth detection for the element attribute to visualize.
       
        Attributes:
            _op_name (str): The operator's name.
            _op_type (OperatorType): The type of the operator.
            _config (dict): Configuration dictionary containing pred_key, eval_key, and el_key.
            _result: The generated 3D scatter plot figure.
       
        Properties:
            op_name (str): Returns the operator's name.
            op_type (OperatorType): Returns the operator's type.
            config (dict): Returns the operator's configuration.
            result: Returns the generated plot.
       
        Methods:
            execute(view: fo.DatasetView): Executes the operator on the given dataset view, filters for true positives,
                extracts relevant data, and generates the visualization.
        
    """
    def __init__(
        self, 
        op_name: str, 
        pred_key: str, 
        eval_key: str, 
        el_key: str
    ) -> None:
        super().__init__()
        self._op_name = op_name
        self._op_type = OperatorType.ConfElDistTPVisualizer
        self._config = {"pred_key": pred_key,
                        "eval_key": eval_key, "el_key": el_key}
        self._result = None
    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def op_type(self) -> OperatorType:
        return self._op_type

    @property
    def config(
        self,
    ) -> dict:
        return self._config

    def _get_gt_eq_from_det(self, det, view, eval_key):
        gt_eq_id = det[f"{eval_key}_id"]

        for sample in view:
            for gt in sample["ground_truth"]["detections"]:
                if gt["id"] == gt_eq_id:
                    return gt
        return None

    @staticmethod
    def _plot_3d_scatter(data, title="3D Scatter Plot"):
        z_data, y_data, x_data = zip(*data)
        # Create 3D scatter plot
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=z_data,  # You can specify a color scale based on the z values
                        colorscale="Viridis",  # Choose a colorscale
                        opacity=0.8,
                    ),
                )
            ]
        )

        # Set axis labels
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X-axis"),
                yaxis=dict(title="Y-axis"),
                zaxis=dict(title="Z-axis"),
            )
        )

        # Set plot title
        fig.update_layout(title=title)

        # Show the plot
        return fig

    def execute(self, view: fo.DatasetView):
        tp_view = view.filter_labels(
            f"{self.config['pred_key']}", F(self.config["eval_key"]) == "tp"
        ).filter_labels("ground_truth", F("label").is_in(["SA06"]))

        conf_el_dist_triplets = []

        for sample in tp_view:
            for det in sample[self.config["pred_key"]]["detections"]:
                gt_eq = self._get_gt_eq_from_det(
                    det, view, self.config["eval_key"])
                conf_el_dist_triplets.append(
                    (
                        det["confidence"],
                        gt_eq[self.config["el_key"]],
                        gt_eq["dist_to_tgt"],
                    )
                )
        self._result = ConfElDistTPVisualizer._plot_3d_scatter(
            conf_el_dist_triplets)

    @property
    def result(self):
        return self._result


class AreaVisualizer(Hist2DVisualizer):
    """
        Visualizes the count of detections per area bin.
        Inherits from Hist2DVisualizer.
    """
    def __init__(self, 
                 op_name: str, 
                 area_field: str = "area",
                 title: str = "Count per Area Bin", 
                 nbins: int = 10, 
                 show: bool = False):
        super().__init__(
            op_name=op_name,
            show=show,
            nxbins=nbins,
            nybins=nbins,
            xrange=1.0,  
            yrange=1.0,  
            title=title,
            xaxis="Area",
            yaxis="Count",
            legend="Area Bins"
        )
        self.area_field = area_field
        self.nbins = nbins
        self.show = show

    def execute(self, view):
        """
        Args:
            view (fo.DatasetView): The FiftyOne dataset view.
        """
        # Collect all areas from detections
        areas = []
        for sample in view:
            for det in sample["ground_truth"]["detections"]:
                area = getattr(det, self.area_field, None)
                if area is not None:
                    areas.append(area)

        if not areas:
            print("[WARNING] No area data found.")
            return

        # Compute bins
        min_area, max_area = min(areas), max(areas)
        bins = np.linspace(min_area, max_area, self.nbins + 1)
        counts, _ = np.histogram(areas, bins=bins)

        # Prepare bin centers for x-axis
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot using Plotly Bar
        layout = go.Layout(
            title=self._config["title"],
            xaxis=dict(title="Area"),
            yaxis=dict(title="Count"),
            legend=dict(title="Area Bins"),
        )
        fig = go.Figure(data=[go.Bar(
            x=bin_centers,
            y=counts,
            marker_color='orange'
        )], layout=layout)

        self._result = fig

        if self.show:
            os.makedirs("./plots", exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            html_path = f"./plots/count_per_area_bins_{timestamp}_plot.html"
            fig.write_html(html_path)
            print(f"[INFO] Plot saved to {html_path}")
            file_uri = Path(html_path).resolve().as_uri()
            webbrowser.open(file_uri)

    @property
    def result(self):
        return self._result


class SqrtAreaVisualizer(Hist2DVisualizer):
    """
        Visualizes the count of detections per sqrt(area) bin.
        Inherits from Hist2DVisualizer.
    """
    def __init__(self, 
                 op_name: str, 
                 area_field: str = "area",
                 title: str = "Count per Sqrt(Area) Bin", 
                 nbins: int = 10, 
                 show: bool = False):
        super().__init__(
            op_name=op_name,
            show=show,
            nxbins=nbins,
            nybins=nbins,
            xrange=1.0,  
            yrange=1.0,  
            title=title,
            xaxis="Sqrt(Area)",
            yaxis="Count",
            legend="Sqrt(Area) Bins"
        )
        self.area_field = area_field
        self.nbins = nbins
        self.show = show

    def execute(self, view):
        """
        Args:
            view (fo.DatasetView): The FiftyOne dataset view.
        """
        # Collect all sqrt(area) from detections
        sqrt_areas = []
        for sample in view:
            for det in sample["ground_truth"]["detections"]:
                area = getattr(det, self.area_field, None)
                if area is not None and area >= 0:
                    sqrt_areas.append(np.sqrt(area))

        if not sqrt_areas:
            print("[WARNING] No sqrt(area) data found.")
            return

        # Compute bins
        min_val, max_val = min(sqrt_areas), max(sqrt_areas)
        bins = np.linspace(min_val, max_val, self.nbins + 1)
        counts, _ = np.histogram(sqrt_areas, bins=bins)

        # Prepare bin centers for x-axis
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot using Plotly Bar
        layout = go.Layout(
            title=self._config["title"],
            xaxis=dict(title="Sqrt(Area)"),
            yaxis=dict(title="Count"),
            legend=dict(title="Sqrt(Area) Bins"),
        )
        fig = go.Figure(data=[go.Bar(
            x=bin_centers,
            y=counts,
            marker_color='green'
        )], layout=layout)

        self._result = fig

        if self.show:
            os.makedirs("./plots", exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            html_path = f"./plots/sqrt_area_bins_{timestamp}_plot.html"
            fig.write_html(html_path)
            print(f"[INFO] Plot saved to {html_path}")
            file_uri = Path(html_path).resolve().as_uri()
            webbrowser.open(file_uri)

    @property
    def result(self):
        return self._result
    

class BBoxSizeVisualizer(Hist2DVisualizer):
    """
        Visualizes the distribution of bounding box width and height (absolute or normalized) as a 2D histogram.
        Inherits from Hist2DVisualizer.
    """
    def __init__(
        self,
        op_name: str,
        width_field: str = "abs_width",   # or "norm_width"
        height_field: str = "abs_height", # or "norm_height"
        xaxis: str = "BBox Width",
        yaxis: str = "BBox Height",
        title: str = "BBox Size Distribution",
        nxbins: int = 20,
        nybins: int = 20,
        xrange: float = 1000,  
        yrange: float = 1000,  
        show: bool = False
    ):
        super().__init__(
            op_name=op_name,
            show=show,
            nxbins=nxbins,
            nybins=nybins,
            xaxis=xaxis,
            yaxis=yaxis,
            xrange=xrange,
            yrange=yrange,
            title=title,
            legend="BBox Size"
        )
        self.width_field = width_field
        self.height_field = height_field

    def execute(self, view):
        """
        Args:
            view (fo.DatasetView): The FiftyOne dataset view.
        """
        widths = []
        heights = []
        for sample in view:
            for det in sample["ground_truth"]["detections"]:
                width = getattr(det, self.width_field, None)
                height = getattr(det, self.height_field, None)
                if width is not None and height is not None:
                    widths.append(width)
                    heights.append(height)

        if not widths or not heights:
            print("[WARNING] No bbox size data found.")
            return

        # Call parent execute to plot 2D histogram
        super().execute(widths, heights)

    @property
    def result(self):
        return self._result


class IouDistributionVisualizer(Hist2DVisualizer):
    """
    Visualizes the distribution and count of IoU values in bins.
    Inherits from Hist2DVisualizer.
    """
    def __init__(
        self,
        op_name: str,
        iou_field: str = "iou",  # field name for IoU in detection
        title: str = "IoU Distribution",
        nbins: int = 20,
        show: bool = False
    ):
        super().__init__(
            op_name=op_name,
            show=show,
            nxbins=nbins,
            nybins=1,
            xrange=1.0,  # IoU ranges from 0 to 1
            yrange=1.0,
            xaxis="IoU",
            yaxis="Count",
            title=title,
            legend="IoU Bins"
        )
        self.iou_field = iou_field
        self.nbins = nbins
        self.show = show

    def execute(self, view):
        """
        Args:
            view (fo.DatasetView): The FiftyOne dataset view.
        """
        # Collect all IoU values from detections
        ious = []
        for sample in view:
            for det in sample["ground_truth"]["detections"]:
                iou = getattr(det, self.iou_field, None)
                if iou is not None:
                    ious.append(iou)

        if not ious:
            print("[WARNING] No IoU data found.")
            return

        # Compute bins
        bins = np.linspace(0, 1, self.nbins + 1)
        counts, _ = np.histogram(ious, bins=bins)

        # Prepare bin centers for x-axis
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot using Plotly Bar
        layout = go.Layout(
            title=self._config["title"],
            xaxis=dict(title="IoU"),
            yaxis=dict(title="Count"),
            legend=dict(title="IoU Bins"),
        )
        fig = go.Figure(data=[go.Bar(
            x=bin_centers,
            y=counts,
            marker_color='purple'
        )], layout=layout)

        self._result = fig

        if self.show:
            os.makedirs("./plots", exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            html_path = f"./plots/iou_distribution_{timestamp}_plot.html"
            fig.write_html(html_path)
            print(f"[INFO] Plot saved to {html_path}")
            file_uri = Path(html_path).resolve().as_uri()
            webbrowser.open(file_uri)

    @property
    def result(self):
        return self._result


class MultipleFiguresPlotter(Operator):
    """
        MultipleFiguresPlotter is an Operator for visualizing multiple Plotly figures (such as histograms) in a single composite figure, arranged according to a specified layout.
            op_name (str): The name of the operator.
            xrange (tuple, optional): The range of x-values to include in the histograms. Default is None.
            yrange (tuple, optional): The range of y-values to include in the histograms. Default is None.
            placement_type (str, optional): The arrangement of the figures. Options are "horizontally", "vertically", or "grid". Default is "horizontally".
            show (bool, optional): Whether to display the resulting figure immediately. Default is False.
            vertical_spacing (float, optional): Vertical spacing between subplots when arranged vertically. Default is 0.3.
            figs_height (int, optional): Height of each subplot in pixels. Default is 400.
            rows (int, optional): Number of rows for grid arrangement. Default is 1.
            cols (int, optional): Number of columns for grid arrangement. Default is 3.
        
        Attributes:
            op_name (str): The name of the operator.
            op_type (OperatorType): The type of the operator.
            config (dict): The configuration dictionary for the operator.
            result (go.Figure): The resulting Plotly figure after execution.
        
        Methods:
            plot_horizontally(figures): Arranges the input figures side by side in a single row.
            plot_vertically(figures): Arranges the input figures in a single column.
            plot_grid(figures): Arranges the input figures in a grid layout based on specified rows and columns.
            execute(figures): Arranges the input figures according to the placement_type and stores the result.
                
    """

    def __init__(
        self,
        op_name: str,
        **kwargs
    ) -> None:
        self._op_name = op_name
        self._op_type = OperatorType.MultipleFiguresPlotter
        self._config = {
            "xrange": kwargs.get('xrange', None),
            "yrange": kwargs.get('yrange', None),
            "placement_type": kwargs.get('placement_type', "horizontally"),
            "show": kwargs.get('show', False),
            "vertical_spacing": kwargs.get('vertical_spacing', 0.3),
            "figs_height": kwargs.get('figs_height', 400),
            "rows": kwargs.get("rows", 1),
            "cols": kwargs.get("cols", 3)
        }
        self._result = None
        
        self._palette = itertools.cycle(px.colors.qualitative.Set2)
        self._class_colors = {}

    @property
    def op_name(self):
        """str: The name of the operator."""
        return self._op_name

    @property
    def op_type(self):
        """OperatorType: The type of the operator."""
        return self._op_type

    @property
    def config(
        self,
    ):
        """dict: The configuration of the operator."""
        return self._config

    # def _get_color(self, idx):
    #     # Use a color palette (extend as needed)
    #     palette = [
    #         "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"
    #     ]
    #     return palette[idx % len(palette)]
    
    def get_color(self, class_name: str) -> str:
        """Return a consistent color for a given class name."""
        if class_name not in self._class_colors:
            self._class_colors[class_name] = next(self._palette)
        return self._class_colors[class_name]
    
    def apply_colors(self, fig):
        """Apply consistent colors to a Plotly figure."""
        for trace in fig["data"]:
            if hasattr(trace, "name") and trace.type in ["bar", "scatter"]:
                trace.marker = getattr(trace, "marker", {})
                trace.marker["color"] = self.get_color(trace.name)
        return fig
    
    def reset_colors(self):
        """Reset the color mapping (fresh palette)."""
        self._class_colors.clear()
        self._palette = itertools.cycle(px.colors.qualitative.Set2)

    # def plot_horizontally(self, figures):
    #     from plotly.subplots import make_subplots

    #     res_fig = make_subplots(
    #         rows=1,
    #         cols=len(figures),
    #         subplot_titles=[fig["layout"]["title"]["text"] for fig in figures],
    #     )

    #     for i, fig in enumerate(figures, start=1):
    #         xaxis_title = fig["layout"]["xaxis"]["title"]["text"]
    #         yaxis_title = fig["layout"]["yaxis"]["title"]["text"]
    #         color = self._get_color(i - 1)

    #         for trace in fig["data"]:
    #             # Only set marker color for Bar or Scatter
    #             if trace.type in ["bar", "scatter"]:
    #                 trace.marker = getattr(trace, "marker", {})
    #                 trace.marker["color"] = color
    #             res_fig.add_trace(trace, row=1, col=i)

    #         res_fig.update_xaxes(title_text=xaxis_title, row=1, col=i)
    #         res_fig.update_yaxes(title_text=yaxis_title, row=1, col=i)

    #     if self._config["xrange"]:
    #         res_fig.update_xaxes(range=self._config["xrange"])
    #     if self._config["yrange"]:
    #         res_fig.update_yaxes(range=self._config["yrange"])

    #     res_fig.update_layout(yaxis=dict(domain=[0, 1]))

    #     return res_fig
    
    def plot_horizontally(self, figures):
        res_fig = make_subplots(
            rows=1,
            cols=len(figures),
            subplot_titles=[fig["layout"]["title"]["text"] for fig in figures],
        )

        for i, fig in enumerate(figures, start=1):
            fig = self.apply_colors(fig)

            xaxis_title = fig["layout"]["xaxis"]["title"]["text"]
            yaxis_title = fig["layout"]["yaxis"]["title"]["text"]

            for trace in fig["data"]:
                res_fig.add_trace(trace, row=1, col=i)

            res_fig.update_xaxes(title_text=xaxis_title, row=1, col=i)
            res_fig.update_yaxes(title_text=yaxis_title, row=1, col=i)

        return res_fig

    def plot_vertically(self, figures):
        from plotly.subplots import make_subplots

        num_rows = len(figures)
        vertical_spacing = min(0.05, 1.0 / max(1, num_rows - 1))  # <-- safe vertical spacing

        res_fig = make_subplots(
            rows=num_rows,
            cols=1,
            subplot_titles=[fig["layout"]["title"]["text"] for fig in figures],
            vertical_spacing=vertical_spacing,
        )

        res_height = 0
        for i, fig in enumerate(figures, start=1):
            xaxis_title = fig["layout"]["xaxis"]["title"]["text"]
            yaxis_title = fig["layout"]["yaxis"]["title"]["text"]
            color = self._get_color(i - 1)
            res_height = res_height + self._config['figs_height'] + vertical_spacing

            for trace in fig["data"]:
                if trace.type in ["bar", "scatter"]:
                    trace.marker = getattr(trace, "marker", {})
                    trace.marker["color"] = color
                res_fig.add_trace(trace, row=i, col=1)

            res_fig.update_xaxes(title_text=xaxis_title, row=i, col=1)
            res_fig.update_yaxes(title_text=yaxis_title, row=i, col=1)

        res_fig.update_layout(height=res_height)

        return res_fig

    def plot_grid(self, figures):
        from plotly.subplots import make_subplots

        res_fig = make_subplots(
            rows=self._config["rows"],
            cols=self._config["cols"],
            subplot_titles=[fig["layout"]["title"]["text"] for fig in figures],
        )

        for i, fig in enumerate(figures, start=1):
            xaxis_title = fig["layout"]["xaxis"]["title"]["text"]
            yaxis_title = fig["layout"]["yaxis"]["title"]["text"]
            color = self._get_color(i - 1)

            row_num = (i - 1) // self._config["cols"] + 1
            col_num = (i - 1) % self._config["cols"] + 1

            for trace in fig["data"]:
                trace.marker = getattr(trace, "marker", {})
                trace.marker["color"] = color
                res_fig.add_trace(trace, row=row_num, col=col_num)

            res_fig.update_xaxes(title_text=xaxis_title, row=row_num, col=col_num)
            res_fig.update_yaxes(title_text=yaxis_title, row=row_num, col=col_num)

        res_fig.update_layout(
            showlegend=False,
            height=400 * self._config["rows"],
            width=600 * self._config["cols"],
            title_text="Grid View",
            title_x=0.5,
            title_y=0.95,
        )

        return res_fig
    def plot_overlay(self, figures):
        """
        Overlay all figures into a single plot (shared x and y axis).
        Useful for comparing multiple histograms directly.
        """
        import plotly.graph_objects as go

        res_fig = go.Figure()

        for i, fig in enumerate(figures):
            color = self.get_color(i)

            for trace in fig["data"]:
                trace.marker = getattr(trace, "marker", {})
                trace.marker["color"] = color
                res_fig.add_trace(trace)

        # Copy axis titles from the first figure
        if figures:
            res_fig.update_xaxes(title_text=figures[0]["layout"]["xaxis"]["title"]["text"])
            res_fig.update_yaxes(title_text=figures[0]["layout"]["yaxis"]["title"]["text"])

        res_fig.update_layout(
            barmode="stack",  # <-- key: stack bars on top of each other
            title_text="Overlay Plot",
            title_x=0.5,
        )

        return res_fig


    def execute(self, figures, step: str = "General") -> None:
        """
            place the figures in positions.

            Args:
                figures: The figures to place beside.

            Returns:
                A plotly figure containing the histograms.
        """
        if self._config["placement_type"] == "horizontally":
            fig = self.plot_horizontally(figures=figures)
        elif self._config["placement_type"] == "vertically":
            fig = self.plot_vertically(figures=figures)
        elif self._config["placement_type"] == "grid":
            fig = self.plot_grid(figures=figures)
        elif self._config["placement_type"] == "overlay":
            fig = self.plot_overlay(figures)    

        if self._config["show"]:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                html_path = f"./plots/{self._op_name}_{timestamp}_{step}_plot.html"
                fig.write_html(html_path)
                print(f"[INFO] Plot saved to {html_path}")
                file_uri = Path(html_path).resolve().as_uri()
                webbrowser.open(file_uri)

        self._result = fig

    @property
    def result(self):
        """go.Figure: The result of the operator."""
        return self._result
    
    
class HistDetectionsAreaSqrtAreaByClass(Operator):
    """
    Plot stacked histogram of detection counts by class,
    binned by area or sqrt(area).

    Args:
        op_name (str): Name of the operator
        bins (int, optional): number of bins. Defaults to 10
        sqrt_area (bool, optional): if True, use sqrt(area). Defaults to True
        orientation (str, optional): "v" for vertical or "h" for horizontal bars. Defaults to "v"
        title (str, optional): Chart title
        xaxis (str, optional): X-axis label
        yaxis (str, optional): Y-axis label
        legend (str, optional): Legend title
        show (bool, optional): Display the chart. Defaults to False
    """

    def __init__(
        self,
        op_name: str,
        bins: int = 10,
        sqrt_area: bool = True,
        orientation: str = "v",
        title: str = "Detection Sqrt Area Distribution by Class",
        xaxis: str = "Sqrt Area range",
        yaxis: str = "Count",
        legend: str = "Class",
        show: bool = False,
    ) -> None:
        self._op_name = op_name
        self._op_type = OperatorType.HistDetectionsSqrtAreaByClass
        self._config = {
            "bins": bins,
            "sqrt_area": sqrt_area,
            "orientation": orientation,
            "title": title,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "legend": legend,
            "show": show,
        }
        self._result = None

    @property
    def op_name(self): return self._op_name
    @property
    def op_type(self): return self._op_type
    @property
    def config(self): return self._config
    @property
    def result(self): return self._result

    def execute(self, view: fo.DatasetView) -> None:
        # Collect detection areas grouped by label
        class_areas = {}
        print("Collecting detection areas per class...")
        for sample in tqdm(view, desc="Processing samples", unit="sample"):
            if sample.ground_truth is None:
                continue
            for det in sample.ground_truth.detections:
                if det.area is None:
                    continue
                area_val = np.sqrt(det.area) if self._config["sqrt_area"] else det.area
                class_areas.setdefault(det.label, []).append(area_val)

        # All areas (for global bin edges)
        all_areas = [a for areas in class_areas.values() for a in areas]
        if not all_areas:
            raise ValueError("No detections with `area` found in dataset")

        bins = self._config["bins"]
        bin_edges = np.linspace(min(all_areas), max(all_areas), bins + 1)
        bin_labels = [
            f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
            for i in range(len(bin_edges) - 1)
        ]

        # Count detections in bins for each class
        bar_segments = []
        print("Building stacked histogram...")
        for label, areas in tqdm(class_areas.items(), desc="Processing classes", unit="class"):
            counts, _ = np.histogram(areas, bins=bin_edges)

            if self._config["orientation"] == "v":  # vertical bars
                bar_segments.append(go.Bar(
                    name=label,
                    x=bin_labels,
                    y=counts
                ))
            else:  # horizontal bars
                bar_segments.append(go.Bar(
                    name=label,
                    y=bin_labels,
                    x=counts,
                    orientation="h"
                ))

        # Create stacked histogram
        fig = go.Figure(data=bar_segments)
        fig.update_layout(
            barmode="stack",
            title=self._config["title"],
            xaxis_title=self._config["xaxis"],
            yaxis_title=self._config["yaxis"],
            legend_title=self._config["legend"],
        )

        self._result = fig

        if self._config["show"]:
            os.makedirs("./plots", exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            html_path = f"./plots/hist_detections_{timestamp}_plot.html"
            fig.write_html(html_path)
            print(f"[INFO] Plot saved to {html_path}")
            file_uri = Path(html_path).resolve().as_uri()
            webbrowser.open(file_uri)

    @property
    def result(self):
        """go.Figure: The result of the operator."""
        return self._result


