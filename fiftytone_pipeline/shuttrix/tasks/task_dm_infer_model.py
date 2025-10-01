import argparse
from tqdm import tqdm
from rich import print

import torch
import torchvision
from ultralytics import YOLO

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

import fiftyone as fo
from shuttrix.tasks.utils import parse_print_args, load_model, run_inference


def init():
    """
    Initializes and parses command-line arguments for running a model on a FiftyOne dataset.
    """
    parser = argparse.ArgumentParser(description="Run model inference on a FiftyOne dataset.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="Name of the FiftyOne dataset to process.")
    
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="Path to the trained model file.")
    
    parser.add_argument('--model_type',
                        type=str,
                        default="yolo",
                        choices=["yolo", "torchvision", "detectron2", "custom"],
                        help="Type of model to load.")
    
    parser.add_argument('--dm_name',
                        type=str,
                        required=True,
                        help="The name of the field to store the predictions (pred_field_name = {dm_name}_pred).")
    
    parser.add_argument('--dm_eval_key',
                        type=str,
                        required=False,
                        help="The name of the field that stores evaluation results (default = {dm_name}_eval).")
    
    parser.add_argument('--only_inf',
                        action='store_true',
                        default=False,
                        help="If set, only run inference.")
    
    parser.add_argument('--only_eval',
                        action='store_true',
                        default=False,
                        help="If set, only run evaluation.")
    
    args = parser.parse_args()
    return args


def load_inference_model(model_type, model_path):
    """
    Loads the model based on the model type.
    """
    if model_type == "yolo":
        return fo.load_model(  # FiftyOne's wrapper
            name="yolo_model",
            model_cls=YOLO,
            model_kwargs={"model": model_path},
            type="detection",
        )
    elif model_type == "torchvision":
        # Example: Faster R-CNN
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()

        return fo.PyTorchModel(model, type="detection")
    elif model_type == "detectron2":
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            predictor = DefaultPredictor(cfg)

            return fo.Detectron2Model(predictor)
        except ImportError as e:
            raise ImportError(f"Detectron2 is not installed. Please install it with: pip install 'git+https://github.com/facebookresearch/detectron2.git'. Error: {e}")
    elif model_type == "custom":
        # User should implement their own loader
        model = load_model(model_path)
        return fo.PyTorchModel(model, type="detection")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


@parse_print_args
def run(ds_name: str, 
        model_path: str, 
        model_type: str,
        dm_name: str,
        dm_eval_key: str,
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
            
    # Load the model (flexible)
    model = load_inference_model(model_type, model_path)

    # Prediction field name
    pred_field_name = f"{dm_name}_pred"
    
    # inference or evaluate model 
    if only_inf:
        print("[green][bold]ONLY INFERENCE MODEL")
        dataset.apply_model(model, pred_field_name)
    elif only_eval:
        print("[green][bold]ONLY EVALUATE MODEL")
        dataset.evaluate_detections(pred_field_name, gt_field="ground_truth", eval_key=dm_eval_key)
    else:
        print("[green][bold]INFERENCE and EVALUATION")
        dataset.apply_model(model, pred_field_name)
        dataset.evaluate_detections(pred_field_name,
                                    gt_field="ground_truth",
                                    eval_key=dm_eval_key)
        
    print(f"\n✅ Predictions added to dataset '{ds_name}' in field '{pred_field_name}'.")
    print(f"Dataset now has {len(dataset)} samples with predictions.")

    # Print evaluation report
    results = dataset.evaluate_detections(pred_field_name, gt_field="ground_truth", eval_key=dm_eval_key)
    counts = dataset.count_values("ground_truth.detections.label")
    classes = sorted(counts, key=counts.get, reverse=True)
    results.print_report(classes=classes)


if __name__ == "__main__":
    args = init()
    run(args.ds_name, 
        args.model_path,
        args.model_type, 
        args.dm_name,
        args.dm_eval_key,
        args.only_inf,
        args.only_eval)