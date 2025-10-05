import cv2
import math
import numpy as np
import argparse
from rich import print

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from shuttrix.operators.field_generator import Filter
from shuttrix.operators.visualizer import Hist2DVisualizer, MultipleFiguresPlotter, ConfusionMatrixSameSampleVisualizer
from shuttrix.tasks import task_dataset_area_sqrtarea_by_column
from shuttrix.tasks.utils import get_area_counts, get_pixel_sizes, parse_print_args


def init():
    """
        This function initializes the task for analyzing a FiftyOne dataset.
        It sets up the command-line argument parser to accept the dataset name.
        
    """
    parser = argparse.ArgumentParser(description="Analyze FiftyOne dataset.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    parser.add_argument('--step',
                        type=str,
                        default="General",
                        help="step name for this task, default: General")
    
    parser.add_argument("--show",
                        type=bool,
                        default=False,
                        help="show plots output.")
    
    args = parser.parse_args()
    return args    
 
 
@parse_print_args
def run(ds_name: str, 
        step: str,
        show: bool,
        splits = ["train", "val", "test"]):
    """
        This function demonstrates the usage of the Filter operator to filter a FiftyOne dataset
        
    """
    print("============================================= ANALYZE DATASET ============================================")
    # load dataset
    dataset = fo.load_dataset(ds_name)  

    labels = dataset.distinct("ground_truth.detections.label")
    print("Labels in dataset: ", labels)

    print(
            """
            =======================================================================================================
            
                                                ANALYZE LABELS COUNT IN DATASET
                                            
            =======================================================================================================
            """ 
    )
    # for whole dataset
    xdata = [] # label names
    ydata = [] # sample counts

    # number of each label in dataset
    for label in labels:
        filtered_data = Filter(
            op_name="Filter_dataset",
            classes=[label],
            splits=splits
        )
        filtered_data.execute(dataset)
        count = len(filtered_data.result)

        xdata.append(label)       
        ydata.append(count)

    hist2d = Hist2DVisualizer(
        op_name="Hist2D_Of_Label_Count",
        nxbins=len(xdata),
        nybins=max(ydata),
        xrange=len(xdata),
        yrange=max(ydata),
        xaxis="Class Label",
        yaxis="Number of Samples",
        title=f"Class Frequency Histogram (Hacked 2D) for Dataset {step}",
        legend="Classes",
        show=show,
        step=step,
        use_hist2d=False
    )
    hist2d.execute(xdata=xdata, ydata=ydata)

    # for each splits
    for split in splits:
        print(f"Split: {split}")
        split_dataset = dataset.filter_labels("ground_truth", fo.ViewField("detections").exists())
        print(f"Number of samples in split '{split}': {len(split_dataset)}")
        
        xdata = [] # label names
        ydata = [] # sample counts
        
        for label in labels:
            filtered_data = Filter(
                op_name="Filter_dataset",
                classes=[label],
                splits=[split]
            )
            filtered_data.execute(dataset)
            count = len(filtered_data.result)

            xdata.append(label)
            ydata.append(count)
            print(f"  - {label}: {count}")
        
        filtered_data = Filter(
            op_name="Filter_dataset",
            classes=[label],
            splits=[f"{split}"]
        )
        filtered_data.execute(dataset)
        count = len(filtered_data.result)

        xdata.append(label)       
        ydata.append(count)
        
        hist2d_each_split = Hist2DVisualizer(
                op_name="Hist2D_Of_Label_Count_In_Each_Split",
                nxbins=len(xdata),
                nybins=max(ydata),
                xrange=len(xdata),
                yrange=max(ydata),
                xaxis="Class Label",
                yaxis="Number of Samples",
                title=f"Class Frequency Histogram (Hacked 2D) for split '{split}' '{step}'",
                legend="Classes",
                show=show,
                split=split,
                step=step,
                use_hist2d=False
        )
        hist2d_each_split.execute(xdata=xdata, ydata=ydata)
        
        
    print(
            """
            =======================================================================================================
            
                                                ANALYZE AREA COUNT IN DATASET
                                            
            =======================================================================================================
            """ 
    )
    # for whole dataset
    xdata, ydata = get_area_counts(dataset)
    
    hist2d_area_count = Hist2DVisualizer(
            op_name="Hist2D_Of_Area_Count",
            nxbins=len(xdata),
            nybins=max(ydata),
            xrange=len(xdata),
            yrange=max(ydata),
            xaxis="Area Count",
            yaxis="Number of Samples",
            title=f"Area Count Histogram (Hacked 2D) for Dataset {step} {step}",
            legend="Area Counts",
            show=True,
            step=step,
            use_hist2d=False
        )    
    hist2d_area_count.execute(xdata, ydata)
    
    figs = []
    for split in splits:
        print(f"compute area for split {split}...")
        filtered_data = dataset.match_tags(f"{split}")
        xdata, ydata = get_area_counts(filtered_data)
        
        if not xdata or not ydata:
            print(f"[WARNING] No samples found in split '{split}'. Skipping plot.")
            continue  # Skip this split
    
        hist2d_area_count_per_Split = Hist2DVisualizer(
                op_name="Hist2D_Of_Area_Count_Per_Split",
                nxbins=len(xdata),
                nybins=max(ydata),
                xrange=len(xdata),
                yrange=max(ydata),
                xaxis="Area Count",
                yaxis="Number of Samples",
                title=f"Area Count Histogram (Hacked 2D) for Split {split} {step}",
                legend="Area Counts",
                show=show,
                step=step,
                use_hist2d=False
            )    
        hist2d_area_count_per_Split.execute(xdata, ydata)
        fig = hist2d_area_count_per_Split.result
        figs.append(fig)
        
    multiplot_area_per_split = MultipleFiguresPlotter(
               op_name="multiplot_area_per_split",
               placement_type="horizontally",
               show=show,                      # show the figure immediately
               rows=1,                         # used for grid mode
               cols=2                          # used for grid mode         
            )
    multiplot_area_per_split.execute(figs, step)   
        
    # for each label
    figs_area_per_labels = []
    for label in labels:
        print(f"compute area for label {label}...")
        filtered_data = dataset.filter_labels("ground_truth", F("label") == label)
        xdata, ydata = get_area_counts(filtered_data)
        
        if not xdata or not ydata:
            print(f"[WARNING] No samples found in label '{label}'. Skipping plot.")
            continue  # Skip this split
        
        hist2d_area_count_per_Labels = Hist2DVisualizer(
                op_name="Hist2D_Of_Area_Count_per_Labels",
                nxbins=len(xdata),
                nybins=max(ydata),
                xrange=len(xdata),
                yrange=max(ydata),
                xaxis="Area Count",
                yaxis="Number of Samples",
                title=f"Area Count Histogram (Hacked 2D) for Labels {label} {step}",
                legend="Area Counts",
                show=show,
                step=step,
                use_hist2d=False
            )    
        hist2d_area_count_per_Labels.execute(xdata, ydata)
        fig = hist2d_area_count_per_Labels.result
        figs_area_per_labels.append(fig)
        
    # multiplot pixels per label 
    multiplot_area_per_label = MultipleFiguresPlotter(
               op_name="multiplot_pixel_per_Labels",
               placement_type="horizontally",
               show=show,                      # show the figure immediately
               rows=math.ceil(math.sqrt(len(labels))),    
               cols=math.ceil(math.sqrt(len(labels)))            
            )
    multiplot_area_per_label.execute(figs_area_per_labels, step)    
    
    
    print(
            """
            =======================================================================================================
            
                                                ANALYZE PIXEL COUNT IN DATASET
                                            
            =======================================================================================================
            """ 
    )
    # for whole dataset
    xdata, ydata = get_pixel_sizes(dataset)
    
    hist2d_pixel_count = Hist2DVisualizer(
            op_name="Hist2D_Of_Pixel_Count",
            nxbins=len(xdata),
            nybins=max(ydata),
            xrange=len(xdata),
            yrange=max(ydata),
            xaxis="Pixel Count",
            yaxis="Number of Samples",
            title=f"Pixel Count Histogram (Hacked 2D) for Dataset {step}",
            legend="Pixel Counts",
            show=show,
            step=step,
            use_hist2d=False
        )    
    hist2d_pixel_count.execute(xdata, ydata)
    
    figs_pixel_per_split = []
    # for each split
    for split in splits:
        print(f"compute area for split {split}...")
        filtered_data = dataset.match_tags(f"{split}")
        xdata, ydata = get_pixel_sizes(filtered_data)
        
        if not xdata or not ydata:
            print(f"[WARNING] No samples found in split '{split}'. Skipping plot.")
            continue  # Skip this split
        
        hist2d_pixel_count_per_Split = Hist2DVisualizer(
                op_name="Hist2D_Of_Pixel_Count_per_Split",
                nxbins=len(xdata),
                nybins=max(ydata),
                xrange=len(xdata),
                yrange=max(ydata),
                xaxis="Pixel Count",
                yaxis="Number of Samples",
                title=f"Pixel Count Histogram (Hacked 2D) for Split {split} {step}",
                legend="Pixel Counts",
                show=show,
                step=step,
                use_hist2d=False
            )    
        hist2d_pixel_count_per_Split.execute(xdata, ydata)
        fig = hist2d_pixel_count_per_Split.result
        figs_pixel_per_split.append(fig)
    multiplot_pixel_per_split = MultipleFiguresPlotter(
               op_name="multiplot_pixel_per_split",
               placement_type="horizontally",
               show=show,                      # show the figure immediately
               rows=1,                         
               cols=3                                   
            )
    multiplot_pixel_per_split.execute(figs_pixel_per_split, step)   
    
    # for each label
    figs_pixel_per_labels = []
    for label in labels:
        print(f"compute pixel for label {label}...")
        filtered_data = dataset.filter_labels("ground_truth", F("label") == label)
        xdata, ydata = get_pixel_sizes(filtered_data)
        
        if not xdata or not ydata:
            print(f"[WARNING] No samples found in label '{label}'. Skipping plot.")
            continue  # Skip this split
        
        hist2d_pixel_count_per_Labels = Hist2DVisualizer(
                op_name="Hist2D_Of_Pixel_Count_per_Labels",
                nxbins=len(xdata),
                nybins=max(ydata),
                xrange=len(xdata),
                yrange=max(ydata),
                xaxis="Pixel Count",
                yaxis="Number of Samples",
                title=f"Pixel Count Histogram (Hacked 2D) for Labels {label}",
                legend="Pixel Counts",
                show=show,
                step=step,
                use_hist2d=False
            )    
        hist2d_pixel_count_per_Labels.execute(xdata, ydata)
        fig = hist2d_pixel_count_per_Labels.result
        figs_pixel_per_labels.append(fig)
        
    # multiplot pixels per label 
    multiplot_pixel_per_labels = MultipleFiguresPlotter(
               op_name="multiplot_pixel_per_Labels",
               placement_type="horizontally",
               show=show,                      # show the figure immediately
               rows=math.ceil(math.sqrt(len(labels))),    
               cols=math.ceil(math.sqrt(len(labels)))            
            )
    multiplot_pixel_per_labels.execute(figs_pixel_per_labels, step)        
    
    
    print(
            """
            =======================================================================================================
            
                                            ANALYZE CO_OCCURRENCE MATRIX IN DATASET
                                            
            =======================================================================================================
            """ 
    )
    # Compute co-occurrence matrix
    print("Computing co-occurrence matrix...")
    co_occurrence_matrix_for_dataset = ConfusionMatrixSameSampleVisualizer(
        op_name="co_occurrence_matrix_for_dataset",
        title="Co-occurrence Matrix for Dataset",
        legend="lower right",
        show=show,
        step=step
    )
    co_occurrence_matrix_for_dataset.execute(view=dataset)
    
    # for each split
    figs = []
    for split in splits:
        print(f"Computing co-occurrence matrix for split '{split}'...")
        co_occurrence_matrix_for_split = ConfusionMatrixSameSampleVisualizer(
            op_name=f"co_occurrence_matrix_for_split_{split}",
            title=f"Co-occurrence Matrix for Split {split} {step}",
            legend="lower right",
            split=split,
            show=show,
            step=step
        )
        co_occurrence_matrix_for_split.execute(view=dataset)
        fig = co_occurrence_matrix_for_split.result
        if fig is not None:
            figs.append(fig)    
        # figs.append(fig)
        
    multiplot_co_occurrence_matrix_for_split = MultipleFiguresPlotter(
            op_name="multiplot_co_occurrence_matrix_for_split",
            placement_type="horizontally",
            show=show,                      # show the figure immediately
            rows=1,                         
            cols=len(figs)                                   
            )    
    multiplot_co_occurrence_matrix_for_split.execute(figs, step)
    
    task_dataset_area_sqrtarea_by_column.run(
        ds_name=ds_name,
        show=show,
        splits=["train", "test", "val"],
    )


if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.step,
        args.show)
