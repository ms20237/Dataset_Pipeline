import numpy as np
import argparse
from rich import print

import fiftyone as fo
from fiftyone import ViewField as F

from shuttrix.operators.visualizer import HistDetectionsAreaSqrtAreaByClass, MultipleFiguresPlotter 
from shuttrix.tasks.utils import parse_print_args, compute_area_nbins


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
    
    parser.add_argument("--splits",
                        type=str,
                        nargs="+",
                        default=None,
                        help="visaulize area of split.")
    
    args = parser.parse_args()
    return args   


@parse_print_args
def run(ds_name: str, 
        splits: str | list,
        show: bool,
        nbins: int = None,
        splits_list = ['train', 'val', 'test']
    ): 
    """
        Analyze Sqrt Area FiftyOne dataset.
        Args:
            ds_name (str): dataset name exp:ds1, ds2, ...
            splits (str): splits to visualize separated by comma, ex: "train,val,test"
            show (bool): show plot result
            nbins (int): Number of bins to use for the area histogram
        
        returns:
            None    
    """
    # load dataset
    dataset = fo.load_dataset(ds_name) 
    print(f"✅ Loaded dataset '{ds_name}' with {len(dataset)} samples")
    
    # compute metadata of dataset
    dataset.compute_metadata() 

    labels = dataset.distinct("ground_truth.detections.label")
    print("Labels in dataset: ", labels)
    
    # Convert splits string to list
    splits_list = splits if isinstance(splits, list) else ([s.strip() for s in splits.split(",")] if splits else [])
    
    if nbins is None:    
        nbins = compute_area_nbins(dataset)    
        
    hist_area_whole_dataset = HistDetectionsAreaSqrtAreaByClass(
        op_name="HistDetectionsAreaByClass",
        title=f"Area by Class for whole dataset",
        sqrt_area=False,
        xaxis="Area",
        yaxis="Number of Detections",
        orientation="v",
        show=show,
        bins=nbins,
    )
    hist_area_whole_dataset.execute(dataset)    
        
    hist_sqrt_area_whole_dataset = HistDetectionsAreaSqrtAreaByClass(
        op_name="HistDetectionsSqrtAreaByClass",
        title=f"Sqrt Area by Class for whole dataset",
        sqrt_area=True,
        xaxis="Area",
        yaxis="Number of Detections",
        orientation="v",
        show=show,
        bins=nbins,
    )        
    hist_sqrt_area_whole_dataset.execute(dataset)
    
    figs_area = []
    figs_sqrt_area = []
    for split in splits_list:
        print(f"Processing split: [italic red]{split}")
        view = dataset.match_tags(split)
        
        # Skip empty splits
        if len(view) == 0:
            print(f"[yellow]Skipping split '{split}' (no samples)")
            continue
        
        hist_area_split = HistDetectionsAreaSqrtAreaByClass(
            op_name=f"HistDetectionsAreaByClass_{split}",
            title=f"Area by Class for {split} split",
            sqrt_area=False,
            xaxis="Area",
            yaxis="Number of Detections",
            orientation="v",
            show=show,
            bins=nbins,
        )
        hist_area_split.execute(view)  
        figs_area.append(hist_area_split.result)  
        
        hist_sqrt_area_split = HistDetectionsAreaSqrtAreaByClass(
            op_name=f"HistDetectionsSqrtAreaByClass_{split}",
            title=f"Sqrt Area by Class for {split} split",
            sqrt_area=True,
            xaxis="Area",
            yaxis="Number of Detections",
            orientation="v",
            show=show,
            bins=nbins,
        )        
        hist_sqrt_area_split.execute(view)
        figs_sqrt_area.append(hist_sqrt_area_split.result)

    # print(len(figs_area))
    # multi_plot_area_per_split = MultipleFiguresPlotter(
    #     op_name="Multiplot_Area_Distribution_persplit",
    #     title="Multiplot Area Distribution persplit",
    #     placement_type="overlay",
    #     rows=1,                         
    #     cols=len(figs_area),
    #     show=True,
    # )
    # multi_plot_area_per_split.execute(figs_area)
    
if __name__ == "__main__":
    args = init()
    run(args.ds_name,
        args.splits,
        args.show,
        args.nbins
    )    