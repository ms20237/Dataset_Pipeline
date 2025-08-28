import numpy as np
import argparse
from rich import print

import fiftyone as fo
from fiftyone import ViewField as F


from shuttrix.operators.visualizer import AreaVisualizer, MultipleFiguresPlotter
from shuttrix.tasks.utils import parse_print_args, compute_area_nbins


def init():
    parser = argparse.ArgumentParser(description="Analyze Area FiftyOne dataset.")
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
        show: bool):
    """
        Visualizes the distribution of detection areas in a FiftyOne dataset.
        Args:
            ds_name (str): The name of the FiftyOne dataset to load.
            nbins (int): Number of bins to use for the area histogram.
            split (str): Dataset split to filter by (e.g., "train", "val", "test"). If None, uses the whole dataset.
            show (bool): Whether to display the histogram visualization.
        Returns:
            None
            
    """
    # load dataset
    dataset = fo.load_dataset(ds_name)
    print(f"✅ Loaded dataset '{ds_name}' with {len(dataset)} samples")  

    labels = dataset.distinct("ground_truth.detections.label")
    print("Labels in dataset: ", labels)
    
    # splits = ["train", "val", "test"]
    # filter splits
    if split is not None:
        view = dataset.match_tags(split)
    else:
        view = dataset    
    
    if nbins is None:
        nbins = compute_area_nbins(view, area_field="area")
        
    dataset_area = AreaVisualizer(
        op_name="Hist Area dataset",
        title="Histogram Area for whole dataset",
        area_field="area",
        nbins=nbins,
        show=show,
    )     
    dataset_area.execute(view)
    
    figs = []
    for label in labels:
        print(f"compute bbox for label {label}...")
        filtered_view = view.filter_labels("ground_truth", F("label") == label)

        if nbins is None:
            nbins = compute_area_nbins(filtered_view, area_field="area")
        filtered_view_area = AreaVisualizer(
            op_name="Hist Area dataset",
            title=f"Histogram Area for label {label}",
            area_field="area",
            nbins=nbins,
            show=show,
        )     
        filtered_view_area.execute(filtered_view)
        figs.append(filtered_view_area.result)
        
    # Multiplot Area per class
    multiplot_area_per_class = MultipleFiguresPlotter(
        op_name="Multiplot Area per class",
        title="Multiplot Area per class",
        show=True,
    )    
    multiplot_area_per_class.execute(figs)
                 

if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.nbins,
        args.split,
        args.show,
    )
    
    