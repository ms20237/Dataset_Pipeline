import numpy as np
import argparse
import json
import math
from tqdm import tqdm
import itertools
from rich import print
from pprint import pprint


import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F

from shuttrix.tasks.utils import parse_print_args


def init():
    """
        Initializes the task for comparing dataset similarity.
        Sets up command-line arguments for the dataset name and the path to a JSON file
        containing label remapping information.
        
        Args:
            ds_name (str): Name of the FiftyOne dataset to analyze.
            json_path (str): Path to the JSON file containing label remapping.
    """
    parser = argparse.ArgumentParser(description="Compare dataset similarity.")
    parser.add_argument('--ds_name',
                        type=str,
                        required=True,
                        help="Name of the FiftyOne dataset to analyze.")
    
    parser.add_argument('--threshold',
                        type=float or int,
                        default=0.8,
                        help="Threshold for similarity comparison. Samples with a similarity score above this value will be considered similar. Default is 0.8.")
    
    parser.add_argument('--mode',
                        type=str,
                        required=True,
                        help="""
                                between_splits or whole_dataset 
                                - choose how to compare similarity. 
                                    'between_splits' compares samples between dataset splits, 
                                    'whole_dataset' compares all samples in the dataset.
                            """)
    
    parser.add_argument('--all_models',
                        action='store_true',
                        help="""
                                If set to 'True', the task will list all available models that can be used for similarity comparison.
                                If set to 'False', you need to specify a model using the --model argument.
                            """)
    
    parser.add_argument('--model',
                        required=True,
                        help="choose model for similarity comparison, options: 'clip', 'clip_vit_b32', 'clip_vit_b16', 'clip_vit_l14', 'clip_vit_l14_336', 'clip_vit_h14_224', 'clip_vit_h14_336'")
    
    parser.add_argument('--similarity_tag',
                        type=str,
                        required=True,
                        help="Tag to apply to similar samples. If a sample's similarity score exceeds the threshold, this tag will be added to the sample.")
    
    parser.add_argument('--recompute',
                        action='store_true',
                        help="""
                                    Recompute similarity scores. If set to 'True', the task will recompute similarity scores for all samples in the dataset. 
                                    If set to 'False', it will use existing scores if available.
                                    Note: This will not delete existing tags, so if you want to reset tags, you need to do it manually.
                            """)

    args = parser.parse_args()
    return args


@parse_print_args
def run(ds_name: str, 
        threshold: float, 
        all_models: bool, 
        model: str, 
        similarity_tag: str, 
        mode: str, 
        recompute: bool, 
        embeddings_field: str = "clip_embedding"):
    print(f"""
            =======================================================================================================
            
                                                CALCULATE SIMILARITY IN {ds_name} 
                                            
            =======================================================================================================
            """
    )
    
    # 
    dataset = fo.load_dataset(ds_name)
    if not dataset:
        raise ValueError(f"Dataset '{ds_name}' not found. Please ensure it exists in FiftyOne.")
    
    # Auto-detect split logic
    if "split" in dataset.get_field_schema():
        split_mode = "field"
        splits = dataset.distinct("split")
    elif dataset.distinct("tags"):
        split_mode = "tags"
        splits = dataset.distinct("tags")
    else:
        raise ValueError("No splits found in dataset. Add a 'split' field or tags like 'train', 'val', etc.")
    
    if all_models:
        all_models = foz.list_zoo_models()
        pprint(all_models)   

    # Embeddings (compute once if whole_dataset)
    if mode == "whole_dataset":
        view = dataset
        if recompute or embeddings_field not in view.get_field_schema():
            print("[INFO] Computing embeddings...")
            model = foz.load_zoo_model(model)
            view.compute_embeddings(model, embeddings_field=embeddings_field)
        else:
            print(f"[INFO] Using cached embeddings in '{embeddings_field}'")

        print("[INFO] Computing similarity over whole dataset...")
        results = fob.compute_similarity(
            view,
            brain_key="sim_whole_dataset",
            embeddings_field=embeddings_field,
            threshold=threshold,
        )

        if results:
            print(f"[INFO] Found {len(results)} similar pairs.")
            for s1, s2, _ in tqdm(results, desc="Tagging similar pairs"):
                s1.tags.append(similarity_tag)
                s2.tags.append(similarity_tag)
                s1.save()
                s2.save()
        else:
            print("[INFO] No similar pairs found.")

    # Similarity between splits
    elif mode == "between_splits":
        if len(splits) < 2:
            raise ValueError("Need at least two splits for between_splits mode")

        for s1_name, s2_name in itertools.combinations(splits, 2):
            print(f"[INFO] Comparing: {s1_name} vs {s2_name}")
            view1 = dataset.match({"split": s1_name}) if split_mode == "field" else dataset.match_tags(s1_name)
            view2 = dataset.match({"split": s2_name}) if split_mode == "field" else dataset.match_tags(s2_name)

            # Ensure embeddings exist
            need_embed = recompute or embeddings_field not in dataset.get_field_schema()
            if need_embed:
                print(f"[INFO] Computing embeddings for {s1_name} and {s2_name}...")
                model = foz.load_zoo_model(model)
                dataset.compute_embeddings(model, embeddings_field=embeddings_field)


            results = fob.compute_similarity(
                                        view1,
                                        support_view=view2,
                                        brain_key=f"sim_{s1_name}_vs_{s2_name}",
                                        embeddings_field=embeddings_field,
                                        threshold=threshold,
                                    )

            if results:
                print(f"[INFO] Found {len(results)} similar pairs between {s1_name} and {s2_name}")
                for s1, s2, _ in tqdm(results, desc=f"Tagging {s1_name} vs {s2_name}"):
                    s1.tags.append(similarity_tag)
                    s2.tags.append(similarity_tag)
                    s1.save()
                    s2.save()
            else:
                print(f"[INFO] No similar pairs found between {s1_name} and {s2_name}")

    else:
        raise ValueError("Invalid mode. Use 'whole_dataset' or 'between_splits'.")
    

if __name__ == "__main__":
    args = init()
    run(args.ds_name, args.threshold, args.all_models, args.model, args.similarity_tag, args.mode, args.recompute)
    
    print(f"Dataset '{args.ds_name}' processed successfully with similarity threshold {args.threshold} using model '{args.model}'.")
    print(f"Samples with similarity scores above {args.threshold} have been tagged with '{args.similarity_tag}'.")    
    if args.mode == "between_splits":
        print("Comparing similarity between dataset splits.")
    elif args.mode == "whole_dataset":
        print("Comparing similarity across the entire dataset.")
        