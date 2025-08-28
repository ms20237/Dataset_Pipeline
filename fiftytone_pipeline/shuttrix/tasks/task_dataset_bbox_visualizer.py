import numpy as np
import argparse
from rich import print

import fiftyone as fo
from fiftyone import ViewField as F

from shuttrix.operators.visualizer import BBoxSizeVisualizer, MultipleFiguresPlotter 
from shuttrix.tasks.utils import parse_print_args, compute_bbox_stats


def init():
    parser = argparse.ArgumentParser(description="Analyze Sqrt Area FiftyOne dataset.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
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
        split: str,
        show: bool,
    ): 
    """
        Visualizes bounding box size distributions for a FiftyOne dataset.
        This function loads a FiftyOne dataset by name, optionally filters it by split,
        and computes statistics and visualizations for bounding box absolute and normalized
        sizes. It generates overall visualizations as well as per-label visualizations for
        each unique label in the dataset.
        Args:
            ds_name (str): Name of the FiftyOne dataset to load.
            split (str): Dataset split to filter by (e.g., "train", "val", "test"). If None, uses the entire dataset.
            show (bool): Whether to display the visualizations.
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
    
    nxbins, nybins, xrange, yrange = compute_bbox_stats(view, "bbox_abs_width", "bbox_abs_height")
    # abs values of height and width
    bbox_abs_vis = BBoxSizeVisualizer(
        op_name="BBox ABS Size",
        width_field="bbox_abs_width",   
        height_field="bbox_abs_height", 
        title="BBox ABS Size Distribution",
        xaxis="BBox abs Width",
        yaxis="BBox abs hieght",
        nxbins=nxbins,
        nybins=nybins,
        xrange=xrange,  
        yrange=yrange,
        show=show,
    )
    bbox_abs_vis.execute(view)
    
    nxbins, nybins, xrange, yrange = compute_bbox_stats(view, "bbox_norm_width", "bbox_norm_height")
    # norm values height and width
    bbox_norm_vis = BBoxSizeVisualizer(
        op_name="BBox Norm Size",
        width_field="bbox_norm_width",   
        height_field="bbox_norm_height", 
        title="BBox Norm Size Distribution",
        xaxis="BBox norm Width",
        yaxis="BBox norm hieght",
        nxbins=nxbins,
        nybins=nybins,
        xrange=xrange,  
        yrange=yrange,
        show=show,
    )
    bbox_norm_vis.execute(view)
    
    abs_figs = []
    norm_figs = []
    for label in labels:
        print(f"compute bbox for label {label}...")
        filtered_view = view.filter_labels("ground_truth", F("label") == label)
        
        nxbins, nybins, xrange, yrange = compute_bbox_stats(filtered_view, "bbox_abs_width", "bbox_abs_height")
        # abs values of height and width
        bbox_abs_vis = BBoxSizeVisualizer(
            op_name="BBox ABS Size",
            width_field="bbox_abs_width",   
            height_field="bbox_abs_height", 
            title=f"BBox ABS Size Distribution for label {label}",
            xaxis="BBox abs Width",
            yaxis="BBox abs hieght",
            nxbins=nxbins,
            nybins=nybins,
            xrange=xrange,  
            yrange=yrange,
            show=show,
        )
        bbox_abs_vis.execute(filtered_view)
        abs_figs.append(bbox_abs_vis.result)
        
        nxbins, nybins, xrange, yrange = compute_bbox_stats(filtered_view, "bbox_norm_width", "bbox_norm_height")
        # norm values height and width
        bbox_norm_vis = BBoxSizeVisualizer(
            op_name="BBox Norm Size",
            width_field="bbox_norm_width",   
            height_field="bbox_norm_height", 
            title=f"BBox Norm Size Distribution for label {label}",
            xaxis="BBox norm Width",
            yaxis="BBox norm hieght",
            nxbins=nxbins,
            nybins=nybins,
            xrange=xrange,  
            yrange=yrange,
            show=show,
        )
        bbox_norm_vis.execute(filtered_view)
        norm_figs.append(bbox_norm_vis.result)
    
    multiplot_abs_norm_figs = MultipleFiguresPlotter(
        op_name="Multiplot Abs and Norm figs",
        title="Multiplot Abs and Norm figs",
        placement_type="vertically",
        show=True,
        
    )
    multiplot_abs_norm_figs.execute(abs_figs+norm_figs)

if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.split,
        args.show,
    )
    
    