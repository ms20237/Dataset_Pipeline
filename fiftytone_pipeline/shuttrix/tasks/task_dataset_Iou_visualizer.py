import numpy as np
import argparse
from rich import print

import fiftyone as fo
from fiftyone import ViewField as F

from shuttrix.operators.visualizer import IouDistributionVisualizer, MultipleFiguresPlotter 
from shuttrix.tasks.utils import parse_print_args, compute_iou_nbins


def init():
    parser = argparse.ArgumentParser(description="Analyze Sqrt Area FiftyOne dataset.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    parser.add_argument('--nbins',
                        type=int,
                        default=None,
                        help="Number of bins to use for the area histogram")
    
    parser.add_argument("--show",
                        type=bool,
                        default=False,
                        help="show plot result")
    
    parser.add_argument("--split",
                        type=str,
                        default=None,
                        help="visaulize area of split.")
    
    args = parser.parse_args()
    return args    
 
 
@parse_print_args
def run(ds_name: str, 
        nbins: int,
        split: str,
        show: bool,
    ): 
    """
        Visualizes the Intersection over Union (IoU) distribution for a FiftyOne dataset, both for the entire dataset and for each individual label.
        Args:
            ds_name (str): The name of the FiftyOne dataset to load.
            nbins (int): Number of bins to use for the IoU histogram. If None, the number of bins is computed automatically.
            split (str): The dataset split to visualize (e.g., "train", "val", "test"). If None, uses the entire dataset.
            show (bool): Whether to display the IoU distribution plots.
        Returns:
            None
        Side Effects:
            - Prints the distinct labels present in the dataset.
            - Displays IoU distribution visualizations for the whole dataset and for each label.
            
    """
    # load dataset
    dataset = fo.load_dataset(ds_name)  

    labels = dataset.distinct("ground_truth.detections.label")
    print("Labels in dataset: ", labels)
    
    # splits = ["train", "val", "test"]
    # filter splits
    if split is not None:
        view = dataset.match_tags(split)
    else:
        view = dataset    
    
    if nbins is None:
        nbins = compute_iou_nbins(view, "iou")
        
    # Iou distribution
    Iou_vis = IouDistributionVisualizer(
        op_name="IoU Distribution",
        title="Iou Distribution for whole dataset",
        iou_field="iou",
        nbins=nbins,
        show=show,
    )
    Iou_vis.execute(view)
    
    figs = []
    for label in labels:
        print(f"compute Iou for label {label}...")
        filtered_view = view.filter_labels("ground_truth", F("label") == label)
        
        nbins = compute_iou_nbins(filtered_view, "iou")
        # Iou distribution
        Iou_vis_for_label = IouDistributionVisualizer(
            op_name="IoU Distribution for each label in dataset",
            title=f"Iou Distribution for label {label} in dataset",
            iou_field="iou",
            nbins=nbins,
            show=show,
        )
        Iou_vis_for_label.execute(filtered_view)
        figs.append(Iou_vis_for_label.result)

    multiplot_Iou_per_class = MultipleFiguresPlotter(
        op_name="Multiplot Iou per class",
        title="Multiplot Iou per class",
        show=True,
    )
    multiplot_Iou_per_class.execute(figs)

if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.nbins,
        args.split,
        args.show,
    )
    
    