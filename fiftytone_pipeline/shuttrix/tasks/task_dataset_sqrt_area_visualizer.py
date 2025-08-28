import numpy as np
import argparse
from rich import print

import fiftyone as fo

from shuttrix.operators.visualizer import SqrtAreaVisualizer 
from shuttrix.tasks.utils import parse_print_args


def init():
    parser = argparse.ArgumentParser(description="Analyze Sqrt Area FiftyOne dataset.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    parser.add_argument('--nbins',
                        type=int,
                        default=80,
                        help="step name for this task, default: General")
    
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
        Visualizes the distribution of the square root of detection areas in a FiftyOne dataset.
        Args:
            ds_name (str): The name of the FiftyOne dataset to load.
            nbins (int): Number of bins to use in the histogram visualization.
            split (str): The dataset split to filter by (e.g., "train", "val", "test"). If None, uses the entire dataset.
            show (bool): Whether to display the visualization.
        Returns:
            None
            
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
    
    dataset_area = SqrtAreaVisualizer(
        op_name="Hist Area for whole dataset",
        area_field="area",
        nbins=nbins,
        show=show,
    )     
    dataset_area.execute(view)
    

if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.nbins,
        args.split,
        args.show,
    )
    
    