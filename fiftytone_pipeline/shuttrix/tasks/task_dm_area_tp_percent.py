import numpy as np
import argparse
from tqdm import tqdm

import fiftyone as fo
from shuttrix.tasks.utils import parse_print_args
from shuttrix.operators.visualizer import Hist2DVisualizer, AreaVisualizer, MultipleFiguresPlotter 


def init():
    parser = argparse.ArgumentParser(description="plot Tp percentage in each dataset area.")
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
                        default="area",
                        help="name of label key for plotting.")
    
    parser.add_argument("--nbins",
                        type=int,
                        default=80,
                        help="number of bins for plotting.")
    
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
        show: bool = False):

    print(f"""
    =======================================================================================================
    
                                    PLOT TP PERCENTAGE PER AREA IN {ds_name}
    
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

    all_dets = []
    for sample in tqdm(dataset):
        if not sample.has_field("ground_truth"):
            continue
        for det in sample["ground_truth"]["detections"]:
            # Safely get area
            area = getattr(det, label_key_name, None)
            # Safely get eval status (like yolov11_eval)
            status = det.get_field(eval_key) if det.has_field(eval_key) else None

            if area is not None and status in ["tp", "fn"]:
                all_dets.append({"area": area, "status": status})

    if not all_dets:
        print("❌ No detections with 'area' and evaluation status found.")
        return

    # Extract area values
    areas = np.array([d["area"] for d in all_dets])
    statuses = np.array([d["status"] for d in all_dets])

    # Bin areas
    area_min, area_max = areas.min(), areas.max()
    bins = np.linspace(area_min, area_max, nbins + 1)
    tp_counts, _ = np.histogram(areas[statuses == "tp"], bins=bins)
    fn_counts, _ = np.histogram(areas[statuses == "fn"], bins=bins)

    tp_percentage = np.divide(tp_counts, tp_counts + fn_counts, 
                              out=np.zeros_like(tp_counts, dtype=float), 
                              where=(tp_counts + fn_counts) != 0) * 100

    # Use midpoints of bins for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    print("[INFO] Average TP%:", np.mean(tp_percentage))

    plots = []
    # Plot using Hist2DVisualizer (1D style)
    Tp_plot = Hist2DVisualizer(
        op_name=f"TP_Percentage_Area",
        show=show,
        nxbins=nbins,
        nybins=1,
        xrange=area_max,
        yrange=100,
        xaxis="Detection Area",
        yaxis="TP Percentage (%)",
        title=f"TP Percentage vs Area ({ds_name})",
        legend="TP %",
        use_hist2d=False)
    Tp_plot.execute(bin_centers.tolist(), tp_percentage.tolist())
    plots.append(Tp_plot.result)
    
    # Plot area of dataset
    area_plot = AreaVisualizer(
    op_name="Area_Distribution_Quickstart",
    area_field="area",       # the field inside each detection
    title="Detection Count per Area Bin",
    nbins=nbins,
    show=show)
    area_plot.execute(dataset)
    plots.append(area_plot.result)
    
    multi_plot = MultipleFiguresPlotter(
        op_name="Tp_Percent_Area",
        title="TP percent with Area values",
        placement_type="horizontally",  # or "vertically", "grid", "overlay"
        show=True,
        cols=len(plots)
    )
    multi_plot.execute(plots) 
    
       
if __name__ == "__main__":
    args = init()
    run(args.ds_name,
        args.dm_name,
        args.dm_eval_key,
        args.label_key_name,
        args.nbins,
        args.show)    
    
    