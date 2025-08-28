import argparse 
from tqdm import tqdm
from rich import print

import fiftyone as fo

from shuttrix.tasks.utils import parse_print_args, compute_iou


def init():
    parser = argparse.ArgumentParser(description="Generate Iou field")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="dataset name exp:ds1, ds2, ...")
    
    args = parser.parse_args()
    return args


@parse_print_args
def run(ds_name: str):
    """
        Adds an Intersection over Union (IoU) field to each detection in the 'ground_truth' of a FiftyOne dataset.
        This function loads the specified dataset, adds a new 'iou' field to the schema for each detection,
        computes the IoU of each detection with respect to the first detection in the sample, and saves the updated samples.
        Args:
            ds_name (str): The name of the FiftyOne dataset to process.
        Side Effects:
            - Modifies the dataset by adding a new 'iou' field to each detection in 'ground_truth'.
            - Saves changes to each sample in the dataset.
        Prints:
            - Status messages indicating progress and completion.
            
    """
    print(f"""
            =======================================================================================================
            
                                                GENERATE IOU FIELD IN {ds_name}
                                            
            =======================================================================================================
            """ 
    )
    # load dataset
    dataset = fo.load_dataset(ds_name)
    print(f"✅ Loaded dataset '{ds_name}' with {len(dataset)} samples")

    # Add IoU field to schema
    dataset.add_sample_field("ground_truth.detections.iou", fo.FloatField)

    for sample in tqdm(dataset, desc="Updating samples with IoU"):
        dets = sample["ground_truth"].detections
        # Example: compute IoU with the first detection as reference
        if dets:
            ref_box = dets[0].bounding_box
            for det in dets:
                det.iou = compute_iou(ref_box, det.bounding_box)
        sample["ground_truth"].detections = dets
        sample.save()
        
    print("✅ Done: Iou field added to each sample for filtering.")
    
if __name__ == "__main__":
    args = init()
    run(args.ds_name)    
    

