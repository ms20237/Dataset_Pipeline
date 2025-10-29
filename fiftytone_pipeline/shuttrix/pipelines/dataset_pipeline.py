import argparse 
from rich import print
import os

import fiftyone as fo

from shuttrix.tasks.utils import parse_print_args, get_similarity_config, load_config_from_file
from shuttrix.tasks import(
    task_dataset_unzip_unrar,
    task_dataset_yaml_fix,
    task_dataset_import_one,
    task_dataset_general_analyze,
    task_dataset_remap_labels,
    task_dataset_ratio,
    task_dataset_similarity,
    task_dataset_merge,
    task_dataset_uniform_labels,
    task_dataset_export,
)

# Define the pipeline steps and their corresponding functions
PIPELINE_STEPS = {
    "unzip_unrar": task_dataset_unzip_unrar.run,
    "yaml_fix": task_dataset_yaml_fix.run,
    "import_one": task_dataset_import_one.run,
    "remap_labels": task_dataset_remap_labels.run,
    "ratio": task_dataset_ratio.run,
    "similarity": task_dataset_similarity.run,
    "merge": task_dataset_merge.run,
    "uniform_labels": task_dataset_uniform_labels.run,
    "export": task_dataset_export.run,
}

def init():
    parser = argparse.ArgumentParser(description="Dataset PipeLine")
    parser.add_argument("--pipeline_config_path",
                        type=str,
                        required=False,
                        default="./configs/pipeline_config_path/dataset_pipeline_config.yaml",
                        help="config path of pipeline to run it.")
    
    parser.add_argument("--ds_dir",
                        type=str,
                        required=False,
                        help="directory of datasets storage")
    
    parser.add_argument("--ds_name",
                        type=str,
                        required=False,
                        help="Path to the JSON file containing a list of dataset names to merge. E.g., './configs/my_datasets_to_merge.json'")
    
    parser.add_argument("--ds_format",
                        type=str,
                        choices=["yolov5", "yolov8", "coco", "voc", "kitti", "fiftyone_image_classification"],
                        required=False,
                        default="yolov5", 
                        help="format of your dataset")
    
    parser.add_argument('--tag_splits',
                        action='store_true', 
                        help="If provided, tags samples with their respective splits (e.g., 'train', 'val', 'test'). Defaults to False if not provided.")
    
    parser.add_argument('--overwrite',
                        action='store_true',
                        help="If set, an existing merged dataset with the same name will be deleted before merging.")
    
    parser.add_argument("--label_path",
                        type=str,
                        required=False,
                        help="Path to the JSON file containing label remapping.")
    
    parser.add_argument("--remove_unmapped_labels",
                        action='store_true',
                        help="If set, removes labels from the dataset that are not found in the JSON remapping file.")
    
    parser.add_argument('--ratios',
                        type=float,
                        nargs=3,
                        default=[0.7, 0.15, 0.15],
                        help="""
                                List of ratios to compare dataset splits. For example, '0.7 0.15 0.15' will compare the first 70%, 15%, and 15% of the dataset.
                                (train, val, test). Default is [0.7, 0.15, 0.15].
                            """)
    
    parser.add_argument('--untag_other',
                        action='store_true',
                        help="If set, removes all tags from samples except 'train', 'val', or 'test'.")
    
    parser.add_argument("--sim_threshold_path",
                        type=str,
                        required=False,
                        help="path of datasets simiarity config")
    
    parser.add_argument('--mode',
                        type=str,
                        required=False,
                        help="""
                                between_splits or whole_dataset 
                                - choose how to compare similarity. 
                                    'between_splits' compares samples between dataset splits, 
                                    'whole_dataset' compares all samples in the dataset.
                            """)
    
    parser.add_argument('--model',
                        required=False,
                        help="choose model for similarity comparison, options: 'alexnet-imagenet-torch', 'centernet-hg104-1024-coco-tf2', 'centernet-hg104-512-coco-tf2'")
    
    parser.add_argument('--recompute',
                        action='store_true',
                        help="""
                                Recompute similarity scores. If set to 'True', the task will recompute similarity scores for all samples in the dataset. 
                                If set to 'False', it will use existing scores if available.
                                Note: This will not delete existing tags, so if you want to reset tags, you need to do it manually.
                            """)
    
    parser.add_argument('--similarity_tag',
                        type=str,
                        default="similar_samples",    
                        help="Tag to apply to similar samples. If a sample's similarity score exceeds the threshold, this tag will be added to the sample.")
    
    parser.add_argument('--list_ds_merge',
                        type=str,
                        nargs='+',
                        default=None,
                        help="A list of dataset names to merge. E.g., 'ds1 ds2 ds3'")
    
    parser.add_argument('--merged_name',
                        type=str,
                        default="merged_dataset",
                        help="Name of the merged dataset (default: 'merged_dataset')")
    
    parser.add_argument('--keep_only_split_tags',
                        action='store_true',
                        help="If set, only 'train', 'val', and 'test' tags will be preserved on samples in the merged dataset. All other tags will be removed.")
    
    parser.add_argument('--output_name',
                        type=str,
                        required=False,
                        help="Name of the output uniform dataset. If not provided, the original dataset will be overwritten.")
    
    parser.add_argument('--export_dir',
                        type=str,
                        default="exported_datasets",
                        help="Directory where the exported dataset will be saved. If not provided, the dataset will not be exported.")
    
    parser.add_argument("--output_ds_format",
                        type=str,
                        choices=["yolov5", "yolov8", "coco", "voc", "kitti", "fiftyone_image_classification"],
                        required=False, 
                        default="yolov5",
                        help="format of your dataset to export")
    
    parser.add_argument('--export_ds_splits', 
                        type=str, 
                        default=None,
                        help="Comma-separated list of splits to export (e.g., 'train,val'). If None, all splits are exported.")
    
    parser.add_argument('--output_label_list',
                        type=str,
                        required=False,
                        help="Comma-separated list of class labels to include in the export (e.g., 'cat,dog,car'). If not provided, all labels will be included.")
    
    parser.add_argument("--step",
                        type=str,
                        required=False,
                        choices=list(PIPELINE_STEPS.keys()),
                        help="Start the pipeline from this specific step.")
    
    parser.add_argument("--run_full_pipeline",
                        action='store_true',
                        required=False,
                        help="Run the full pipeline from start to finish.")
    
    args = parser.parse_args()
    return parser, args
    

@parse_print_args    
def run(args):
    """
    Main function to run the dataset pipeline based on command-line arguments.
    """
    
    print("""

            ███████╗██╗  ██╗██╗   ██╗████████╗████████╗██████╗ ██╗██╗  ██╗
            ██╔════╝██║  ██║██║   ██║╚══██╔══╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝
            ███████╗███████║██║   ██║   ██║      ██║   ██████╔╝██║ ╚███╔╝ 
            ╚════██║██╔══██║██║   ██║   ██║      ██║   ██╔══██╗██║ ██╔██╗ 
            ███████║██║  ██║╚██████╔╝   ██║      ██║   ██║  ██║██║██╔╝ ██╗
            ╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
                                            P i p e l i n e (Dataset)  v1.0.0  

    """)
    
    # Determine the starting step of the pipeline
    pipeline_steps_list = list(PIPELINE_STEPS.keys())
    start_step_index = 0
    
    if args.run_full_pipeline:
        print("[bold][green]Running full pipeline from the beginning.[/bold]")
    elif args.step:
        try:
            start_step_index = pipeline_steps_list.index(args.step)
            print(f"[bold][green]Starting pipeline from step: '{args.step}'[/bold]")
        except ValueError:
            raise ValueError(f"Invalid step '{args.step}'. Valid steps are: {list(PIPELINE_STEPS.keys())}")
    else:
        # Default behavior if no specific mode is chosen
        print("[yellow]No pipeline mode specified. Defaulting to running the full pipeline.[/yellow]")
        args.run_full_pipeline = True
    
    
    # --- Pipeline execution based on the starting step ---
    
    # Per-dataset processing steps
    dataset_names_for_merge = []
    
    if start_step_index <= pipeline_steps_list.index("similarity"):
        if args.ds_dir is None:
            raise ValueError("The 'ds_dir' argument is required for these initial steps.")

        if not os.path.exists(args.ds_dir):
            raise ValueError(f"Dataset directory {args.ds_dir} does not exist")
        
        dataset_folders = [f for f in os.listdir(args.ds_dir) 
                          if os.path.isdir(os.path.join(args.ds_dir, f))]
        
        if not dataset_folders:
            print(f"[yellow]Warning: No dataset folders found in {args.ds_dir}[/yellow]")
            return
        
        print(f"[green]Found {len(dataset_folders)} datasets to process:[/green]")
        for folder_name in dataset_folders:
            dataset_name = f"{folder_name}"
            dataset_names_for_merge.append(dataset_name)
            
            ds_path = os.path.join(args.ds_dir, folder_name)

            # TODO: fix this 2 tasks
            # # Check if we should run each step based on the start_step_index
            # if start_step_index <= pipeline_steps_list.index("unzip_unrar"):
            #     # Unzip or unrar the dataset if needed
            #     print(f"[bold]Running 'unzip_unrar' for: {folder_name}[/bold]")
            #     task_dataset_unzip_unrar.run(datasets_dir=args.ds_dir)
            
            # if start_step_index <= pipeline_steps_list.index("yaml_fix"):
            #     # Fix YAML files if needed
            #     print(f"[bold]Running 'yaml_fix' for: {folder_name}[/bold]")
            #     task_dataset_yaml_fix.run(datasets_dir=args.ds_dir)
            
            if start_step_index <= pipeline_steps_list.index("import_one"):
                # Import the dataset
                print(f"[bold]Running 'import_one' for: {folder_name}[/bold]")
                task_dataset_import_one.run(ds_path=ds_path,
                                            ds_name=dataset_name,
                                            format=args.ds_format,
                                            overwrite=args.overwrite,
                                            tag_splits=args.tag_splits)
            
            if start_step_index <= pipeline_steps_list.index("remap_labels"):
                # remapping labels
                remap_json_path = os.path.join(args.label_path, f"{folder_name}.json") if args.label_path else None
                if remap_json_path and os.path.exists(remap_json_path):
                    print(f"[bold][green]Running 'remap_labels' for {dataset_name} using {remap_json_path}[/bold]")
                    task_dataset_remap_labels.run(
                        ds_name=dataset_name,
                        json_path=remap_json_path,
                        remove_unmapped_labels=args.remove_unmapped_labels,
                        new_dataset=False,
                    )
                else:
                    print(f"[yellow]Warning: No remapping file found for '{folder_name}'. Skipping remapping.[/yellow]")
            
            if start_step_index <= pipeline_steps_list.index("ratio"):
                # change ratio of dataset
                print(f"[bold]Running 'ratio' for: {dataset_name}[/bold]")
                task_dataset_ratio.run(
                    ds_name=dataset_name,
                    ratios=args.ratios,
                    untag_other=args.untag_other,
                )
            
            if start_step_index <= pipeline_steps_list.index("similarity"):
                # check similarity 
                if args.sim_threshold_path:
                    do_sim, threshold = get_similarity_config(args.sim_threshold_path, folder_name)
                    if do_sim and threshold is not None:
                        print(f"[bold][green]Running 'similarity' task for {dataset_name} with threshold {threshold}[/bold]")
                        task_dataset_similarity.run(ds_name=dataset_name, 
                                                    threshold=threshold,
                                                    mode=args.mode,
                                                    model=args.model,
                                                    recompute=args.recompute,
                                                    similarity_tag=args.similarity_tag,
                                                    all_models=False)
                else:
                    print("[yellow]Warning: No sim_threshold_path provided. Skipping similarity task.[/yellow]")

    current_ds_name = None
    
    # STEP: Merge datasets
    if start_step_index <= pipeline_steps_list.index("merge"):
        if not dataset_names_for_merge:
             dataset_names_for_merge = [f for f in os.listdir(args.ds_dir) if os.path.isdir(os.path.join(args.ds_dir, f))]

        if args.list_ds_merge is None:
            if len(dataset_names_for_merge) > 1:
                print(f"[bold][green]Running 'merge' for datasets: {dataset_names_for_merge}[/bold]")
                task_dataset_merge.run(in_ds_names=dataset_names_for_merge, 
                                     merged_name=args.merged_name, 
                                     overwrite=args.overwrite, 
                                     keep_only_split_tags=args.keep_only_split_tags)
                current_ds_name = args.merged_name
            else:
                print("[yellow]Only one dataset found. Skipping merge.[/yellow]")
                current_ds_name = dataset_names_for_merge[0] if dataset_names_for_merge else None
        else:
            print(f"[bold][green]Running 'merge' for datasets from list: {args.list_ds_merge}[/bold]")
            task_dataset_merge.run(in_ds_names=args.list_ds_merge, 
                                 merged_name=args.merged_name, 
                                 overwrite=args.overwrite, 
                                 keep_only_split_tags=args.keep_only_split_tags)
            current_ds_name = args.merged_name

    if not current_ds_name:
        if args.ds_name:
             current_ds_name = args.ds_name
        else:
            print("[red]Error: No dataset to process for later steps. Exiting.[/red]")
            return

    # STEP: Uniform labels
    if start_step_index <= pipeline_steps_list.index("uniform_labels"):
        print(f"[bold][green]Running 'uniform_labels' for dataset: {current_ds_name}[/bold]")
        uniform_ds_name = args.output_name if args.output_name else f"{current_ds_name}_uniform"
        task_dataset_uniform_labels.run(ds_name=current_ds_name, 
                                         output_name=uniform_ds_name)
        current_ds_name = uniform_ds_name
        print(f"[bold][green]Uniform labels dataset created successfully as '{current_ds_name}'![/bold]") 
    
    # STEP: Export dataset
    if start_step_index <= pipeline_steps_list.index("export"):
        print(f"[bold][green]Running 'export' for dataset: {current_ds_name}[/bold]")
        task_dataset_export.run(ds_name=current_ds_name, 
                                 output_dir=args.export_dir, 
                                 format=args.output_ds_format,
                                 splits=args.export_ds_splits,
                                 label_list=args.output_label_list)
        print(f"[bold][green]Dataset exported successfully to {args.export_dir}[/bold]")

    print(f"[bold][green]Dataset pipeline completed successfully![/bold]")
    print("[bold][green]PIPELINE COMPLETE!!!![/bold]")
    
if __name__ == "__main__":
    parser, args = init()
    config = {}
    if args.pipeline_config_path:
        print(f"[bold][green]Loading configuration from {args.pipeline_config_path}[/bold]")
        config = load_config_from_file(args.pipeline_config_path)

    final_config = {}
    for action in parser._actions:
        if action.dest != 'help':
            final_config[action.dest] = action.default

    final_config.update(config)

    for key, value in vars(args).items():
        if value is not None and value != parser.get_default(key):
            final_config[key] = value

    args = argparse.Namespace(**final_config)
    
    run(args)