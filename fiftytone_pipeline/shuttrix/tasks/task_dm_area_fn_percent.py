import numpy as np
import argparse
from tqdm import tqdm

import fiftyone as fo
from shuttrix.tasks.utils import parse_print_args
from shuttrix.operators.visualizer import Hist2DVisualizer, AreaVisualizer, MultipleFiguresPlotter 


def init():
    parser = argparse.ArgumentParser(description="plot Fn percentage in each dataset area.")
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
        show: bool):

    print(f"""
    =======================================================================================================
    
                                   PLOT FN PERCENTAGE PER AREA IN {ds_name}
    
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
        width, height = sample.metadata.width, sample.metadata.height

        # Ground Truth detections
        if sample.has_field("ground_truth"):
            for det in sample["ground_truth"]["detections"]:
                # Get area
                area = getattr(det, label_key_name, None)
                if area is None and hasattr(det, "bounding_box"):
                    bbox = det.bounding_box
                    if len(bbox) >= 4:
                        area = bbox[2] * bbox[3]

                # Get evaluation status
                status = det.get_field(eval_key) if det.has_field(eval_key) else None

                if area is not None and status in ["tp", "fn"]:
                    all_dets.append({"area": area, "status": status})

    if not all_dets:
        print("❌ No detections with 'area' and evaluation status found.")
        return

    areas = np.array([d["area"] for d in all_dets])
    statuses = np.array([d["status"] for d in all_dets])

    # Bin the areas
    area_min, area_max = areas.min(), areas.max()
    bins = np.linspace(area_min, area_max, nbins + 1)

    tp_counts, _ = np.histogram(areas[statuses == "tp"], bins=bins)
    fn_counts, _ = np.histogram(areas[statuses == "fn"], bins=bins)

    # Compute FN percentage
    fn_percentage = np.divide(fn_counts, tp_counts + fn_counts,
                              out=np.zeros_like(fn_counts, dtype=float),
                              where=(tp_counts + fn_counts) != 0) * 100

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    print("[INFO] Average FN%:", np.mean(fn_percentage))

    plots = []

    # FN% Plot
    fn_plot = Hist2DVisualizer(
        op_name="FN_Percentage_Area",
        show=show,
        nxbins=nbins,
        nybins=1,
        xrange=area_max,
        yrange=100,
        xaxis="Detection Area",
        yaxis="FN Percentage (%)",
        title=f"FN Percentage vs Area ({ds_name})",
        legend="FN %",
        use_hist2d=False)
    fn_plot.execute(bin_centers.tolist(), fn_percentage.tolist())
    plots.append(fn_plot.result)

    # Area distribution plot
    area_plot = AreaVisualizer(
        op_name="Area_Distribution_FN",
        area_field="area",
        title="Detection Count per Area Bin",
        nbins=nbins,
        show=show)
    area_plot.execute(dataset)
    plots.append(area_plot.result)

    multi_plot = MultipleFiguresPlotter(
        op_name="FN_Percent_Area",
        title="FN percent with Area values",
        placement_type="horizontally",
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
    
    