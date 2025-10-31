import numpy as np
import argparse
from tqdm import tqdm
from pprint import pprint

import fiftyone as fo
from shuttrix.tasks.utils import parse_print_args
from shuttrix.operators.visualizer import PrecisionRecallPerClass, MultipleFiguresPlotter 


def init():
    parser = argparse.ArgumentParser(description="plot Precision/Recall in each dataset bbox.")
    parser.add_argument("--ds_name",
                        type=str,
                        required=True,
                        help="Name of the FiftyOne dataset to process.")
    
    parser.add_argument("--dm_name",
                        type=str,
                        required=True,
                        help="name of model predict field.")
    
    parser.add_argument("--dm_eval_key",
                        type=str,
                        required=False,
                        help="name of model evaluation field.")
    
    parser.add_argument("--label_key_name",
                        type=str,
                        default="bbox",
                        help="name of label key for plotting.")
    
    parser.add_argument("--nbins",
                        type=int,
                        default=80,
                        help="number of bins for plotting.")
    
    parser.add_argument("--step_value",
                        type=float,
                        default=0.01,
                        help="step value of bbox height/width for plotting percision/recall per bbox values")
    
    parser.add_argument("--show",
                        type=bool,
                        default=False,
                        help="show final plot.")
    
    args = parser.parse_args()
    return args  


@parse_print_args
def run(ds_name: str,
        dm_name: str,
        dm_eval_key: str,
        label_key_name: str,
        nbins: int,
        step_value: float,
        show: bool = False):

    print(f"""
    =======================================================================================================
    
                                    PLOT PRECISION/RECALL PER BBOX IN {ds_name}
    
    =======================================================================================================
    """)

    if not fo.dataset_exists(ds_name):
        print(f"❌ Dataset '{ds_name}' not found.")
        return

    dataset = fo.load_dataset(ds_name)
    print(f"✅ Loaded dataset '{ds_name}' with {len(dataset)} samples")

    eval_key = dm_eval_key or f"{dm_name}_eval"
    pred_field = f"{dm_name}_pred"

    print(f"Using eval key: {eval_key}")
    print(f"Using prediction field: {pred_field}")

    # print labels in dataset
    labels = dataset.distinct("ground_truth.detections.label")
    print("📋 Labels in dataset:")
    pprint(labels)

    # Instantiate the visualizer
    visualizer = PrecisionRecallPerClass(
        op_name="PrecisionRecallPerClass",
        show=show,
        dm_eval_key=eval_key,
        title=f"Precision/Recall per Class for {ds_name}",
        step_value=step_value,
        step="BBoxEval",
    )
    visualizer.execute(dataset)
    
    # multiplot Precision/Recall plots
    plotter = MultipleFiguresPlotter(op_name="MultiPlot", 
                                     placement_type="grid", 
                                     rows=4, 
                                     cols=4, 
                                     show=True)
    plotter.execute(visualizer.result)


if __name__ == "__main__":
    args = init()
    run(args.ds_name,
        args.dm_name,
        args.dm_eval_key,
        args.label_key_name,
        args.nbins,
        args.step_value,
        args.show)    
    
    

