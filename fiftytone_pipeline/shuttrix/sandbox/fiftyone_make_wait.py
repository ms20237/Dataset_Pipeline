import fiftyone as fo

# Load saved dataset
dataset = fo.load_dataset("check1")

# Connect to the already-running app
session = fo.launch_app(dataset, remote=True)
