import numpy as np
import argparse
import fiftyone as fo
from shuttrix.tasks.utils import parse_print_args, load_config_from_file
from shuttrix.tasks import(
    task_dm_area_tp_percent,
    task_dm_area_fp_percent,
    task_dm_area_fn_percent,
    task_dm_bbox_precision_recall_perclass,
    task_dm_bbox_tp_fp_fn_perclass,
)


# Define the pipeline steps and their corresponding functions
PIPELINE_STEPS = {
    "plot_tp_area": task_dm_area_tp_percent.run,
    "plot_fp_area": task_dm_area_fp_percent.run,
    "plot_fn_area": task_dm_area_fn_percent.run,
    "plot_precision_recall_perclass": task_dm_bbox_precision_recall_perclass.run,
    "plot_tp_fp_fn_perclass": task_dm_bbox_tp_fp_fn_perclass.run,
}


def init():
    parser = argparse.ArgumentParser(description="Model Analyze PipeLine")
    parser.add_argument("--pipeline_config_path",
                        type=str,
                        required=False,
                        default="./configs/pipeline_config_path/pipeline_model_config.yaml",
                        help="config path of pipeline to run it.")
    
    parser.add_argument("--ds_name",
                        type=str,
                        required=False,
                        help="Path to the JSON file containing a list of dataset names to merge. E.g., './configs/my_datasets_to_merge.json'")
    
    parser.add_argument("--dm_name",
                        type=str,
                        required=False,
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
    return args


@parse_print_args    
def run(args):
    """
    Main function to run the model pipeline based on command-line arguments.
    """
    print("""

            ███████╗██╗  ██╗██╗   ██╗████████╗████████╗██████╗ ██╗██╗  ██╗
            ██╔════╝██║  ██║██║   ██║╚══██╔══╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝
            ███████╗███████║██║   ██║   ██║      ██║   ██████╔╝██║ ╚███╔╝ 
            ╚════██║██╔══██║██║   ██║   ██║      ██║   ██╔══██╗██║ ██╔██╗ 
            ███████║██║  ██║╚██████╔╝   ██║      ██║   ██║  ██║██║██╔╝ ██╗
            ╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
                                        P i p e l i n e (Analyze Model)  v1.0.0  

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
        # Default behavior: run the full pipeline if neither flag is provided
        print("[yellow]No 'step' or 'run_full_pipeline' specified. Running full pipeline by default.[/yellow]")
        args.run_full_pipeline = True
        start_step_index = 0
        
    # --- Pipeline execution based on the starting step ---
    
    # plot tp/fp/fn model on dataset
    if start_step_index <= pipeline_steps_list.index("plot_tp_fp_fn_perclass"):
        if args.ds_name is None:
            raise ValueError("The 'ds_name' argument is required for these initial steps.")
        
        elif args.dm_name is None:
            raise ValueError("The 'dm_name' argument is required for these initial steps.")
        
        if start_step_index <= pipeline_steps_list.index("plot_tp_area"):
                # Plot tp area model
                print(f"[bold]Running 'plot tp area'")
                task_dm_area_tp_percent.run(ds_name=args.ds_name,
                                            dm_name=args.dm_name,
                                            dm_eval_key=args.dm_eval_key,
                                            label_key_name=args.label_key_name,
                                            nbins=args.nbins)

        if start_step_index <= pipeline_steps_list.index("plot_fp_area"):
                # Plot fp area model
                print(f"[bold]Running 'plot fp area'")
                task_dm_area_fp_percent.run(ds_name=args.ds_name,
                                            dm_name=args.dm_name,
                                            dm_eval_key=args.dm_eval_key,
                                            label_key_name=args.label_key_name,
                                            nbins=args.nbins)
                
        if start_step_index <= pipeline_steps_list.index("plot_fn_area"):
                # Plot fn area model
                print(f"[bold]Running 'plot fn area'")
                task_dm_area_fn_percent.run(ds_name=args.ds_name,
                                            dm_name=args.dm_name,
                                            dm_eval_key=args.dm_eval_key,
                                            label_key_name=args.label_key_name,
                                            nbins=args.nbins)
                
        if start_step_index <= pipeline_steps_list.index("plot_precision_recall_perclass"):
                # Plot precision/recall perclass
                print(f"[bold]Running 'plot precision/recall perclass'")
                task_dm_bbox_precision_recall_perclass.run(ds_name=args.ds_name,
                                                            dm_name=args.dm_name,
                                                            dm_eval_key=args.dm_eval_key,
                                                            label_key_name=args.label_key_name,
                                                            nbins=args.nbins,
                                                            step_value=args.step_value)  
                
        if start_step_index <= pipeline_steps_list.index("plot_tp_fp_fn_perclass"):
                # Plot tp_fp_fn perclass
                print(f"[bold]Running 'plot tp_fp_fn perclass'")
                task_dm_bbox_tp_fp_fn_perclass.run(ds_name=args.ds_name,
                                                    dm_name=args.dm_name,
                                                    dm_eval_key=args.dm_eval_key,
                                                    label_key_name=args.label_key_name,
                                                    nbins=args.nbins,
                                                    step_value=args.step_value)                
        
    print(f"[bold][green]Model pipeline completed successfully![/bold]")
    print("[bold][green]PIPELINE COMPLETE!!!![/bold]")        

                
if __name__ == "__main__":
    args = init()
    config = {}

    if args.pipeline_config_path:
        print(f"[bold][green]Loading configuration from {args.pipeline_config_path}[/bold]")
        config = load_config_from_file(args.pipeline_config_path)

    # Merge YAML config with argparse args
    final_config = vars(args).copy()
    final_config.update(config)
    args = argparse.Namespace(**final_config)

    run(args)
                 


    

