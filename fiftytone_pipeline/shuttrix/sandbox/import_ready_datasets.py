import urllib.request

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from shuttrix.operators.visualizer import ConfusionMatrixSameSampleVisualizer

dataset = foz.load_zoo_dataset("quickstart")

# Get all available models
all_models = foz.list_zoo_models()

# Filter models that can produce embeddings (needed for similarity)
embedding_models = []
for model_name in all_models:
    try:
        info = foz.get_zoo_model_info(model_name)
        if info.supports_embeddings:
            embedding_models.append(model_name)
    except:
        pass  # some models might not expose info or raise errors

# Print the embedding models
print("📦 Models that support similarity (embeddings):")
for name in embedding_models:
    print(f"  - {name}")


try:
    urllib.request.urlopen("https://voxel51.com", timeout=5)
    print("Internet is working")
except:
    print("No internet access for model listing")
    
    
# view = dataset.match_tags("train")
# view2 = dataset.filter_labels( "predictions", F("label")== "dog")


# view = dataset.map_labels(
#     "predictions", {"rabbit": "other", "squirrel": "other"}
# )

# print(dataset)
# print(len(view))
# print(len(view2))
# print(len(dataset.filter_field("predictions", F("label") == "cat")))

# This auto-handles multiple labels per sample


# confusion_dataset = ConfusionMatrixSameSampleVisualizer(
#     op_name="confusion_matrix",
#     show=True,
#     title="Confusion Matrix",
#     legend="lower right",
#     )
# confusion_dataset.execute(view=dataset)
# session = fo.launch_app(dataset)
# session.wait()

