import fiftyone as fo

def delete_all_datasets():
    """Delete all datasets in FiftyOne"""
    # Get list of all existing datasets
    dataset_names = fo.list_datasets()
    
    if not dataset_names:
        print("No datasets found in FiftyOne")
        return
    
    print(f"Found {len(dataset_names)} datasets to delete:")
    for name in dataset_names:
        print(f" - {name}")
    
    # Delete each dataset
    for name in dataset_names:
        try:
            fo.delete_dataset(name)
            print(f"Deleted dataset: {name}")
        except Exception as e:
            print(f"Error deleting dataset {name}: {str(e)}")
    
    print("All datasets have been deleted")

if __name__ == "__main__":
    delete_all_datasets()