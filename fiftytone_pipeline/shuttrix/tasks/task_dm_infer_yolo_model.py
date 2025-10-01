import argparse
from tqdm import tqdm
from rich import print

from ultralytics import YOLO

import fiftyone as fo
from shuttrix.tasks.utils import parse_print_args, load_model, run_inference


def init():
    """
    Initializes and parses command-line arguments for running a model on a FiftyOne dataset.
    """
    parser = argparse.ArgumentParser(description="Run model inference YOLO models on FiftyOne dataset.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="Name of the FiftyOne dataset to process.")
    
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="Path to the trained model file.")
    
    parser.add_argument('--dm_name',
                        type=str,
                        required=True,
                        help="The name of the field to store the predictions (pred_field_name = {dm_name}_pred).")
    
    parser.add_argument('--dm_eval_key',
                        type=str,
                        required=False,
                        help="The name of the field that store evaluation of model (defualt pred_field_name = {dm_name}_Eval).")
    
    parser.add_argument('--conf_thresh',
                        type=float,
                        default=0.3,
                        help="confidence threshold for visualize in fiftyone.(values between 0-1)")
    
    parser.add_argument('--only_inf',
                        action='store_true',
                        default=False,
                        help="If set, it will inference model on dataset in fiftyone.")

    parser.add_argument('--only_eval',
                        action='store_true',
                        default=False,
                        help="If set, it will Evaluate model on dataset in fiftyone.")
    
    args = parser.parse_args()
    return args


@parse_print_args
def run(ds_name: str, 
        model_path: str, 
        dm_name: str,
        dm_eval_key: str,
        conf_thresh: float,
        only_inf: bool,
        only_eval: bool):
    """
    Main function to run model inference and store predictions on a dataset.
    """
    print(f"""
            ======================================================================================
            
                                    [bold]RUNNING INFERENCE ON DATASET: {ds_name}
            
            ======================================================================================
    """)

    # Load the FiftyOne dataset
    if not fo.dataset_exists(ds_name):
        print(f"❌ Dataset '{ds_name}' not found. Exiting.")
        return
    dataset = fo.load_dataset(ds_name)
    print(f"✅ Loaded dataset: {ds_name} with {len(dataset)} samples.")

    # set eval_key field
    if dm_eval_key is None:
        dm_eval_key = f"{dm_name}_eval"
    else:
        dm_eval_key = dm_eval_key
            
    # Load the model
    model = YOLO(model_path)

    # Iterate through each sample and run inference
    pred_field_name = f"{dm_name}_pred"
    
    # inference or evaluate model 
    if only_inf:
        print("[green][bold]ONLY INFERENCE MODEL")
        inf_model = dataset.apply_model(model, 
                                        pred_field_name,
                                        confidence_thresh=conf_thresh)
    elif only_eval:
        print("[green][bold]ONLY EVALUATE MODEL")
        eval_model = dataset.evaluate_detections(eval_key=dm_eval_key)
    else:
        print("[green][bold]ONLY INFERENCE and EVALUATE MODEL")
        inf_model = dataset.apply_model(model, 
                                        pred_field_name,
                                        confidence_thresh=conf_thresh)
        eval_model = dataset.evaluate_detections(pred_field_name,
                                    gt_field="ground_truth",
                                    eval_key=dm_eval_key)
        
    print(f"\n✅ Successfully added predictions to dataset '{ds_name}' in field '{pred_field_name}'.")
    print(f"Dataset now has {len(dataset)} samples with {dataset.count(pred_field_name)} predictions.")

    counts = dataset.count_values("ground_truth.detections.label")
    classes = sorted(counts, key=counts.get, reverse=True)
    
    # Print a classification report for the top-10 classes
    eval_model.print_report(classes=classes)


if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.model_path, 
        args.dm_name,
        args.dm_eval_key,
        args.conf_thresh,
        args.only_inf,
        args.only_eval)
