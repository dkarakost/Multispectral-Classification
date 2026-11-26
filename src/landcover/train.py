from landcover.data import load_image, load_labels, get_band, flatten_image, flatten_labels, unflatten_predictions
from landcover.features import add_indices
from landcover.evaluate import macro_iou, per_class_iou
from landcover.models import make_knn,make_rf,make_knn_pca
from landcover.visualization import plot_map,plot_legend,plot_confusion
import matplotlib.pyplot as plt 
import os
import numpy as np



s2_arn = load_image("data/raw/s2_arn.tif")
s2_ren = load_image("data/raw/s2_ren.tif")
s2_wur = load_image("data/raw/s2_wur.tif")
print("The size of the image before the indices is: ", s2_arn.shape)

# add the extra indices to the images 
s2_arn = add_indices(s2_arn)
s2_ren = add_indices(s2_ren)
s2_wur = add_indices(s2_wur)
print("The size of the image after adding the indices is: ", s2_arn.shape)

# flatten the images into a 2D 
X_train = flatten_image(s2_arn)
X_val = flatten_image(s2_ren)
X_wur = flatten_image(s2_wur)
print("The size of the training set with the features is: ",X_train.shape)

# load the labels (Arnhem and Renkum only)
lc_arn = load_labels("data/raw/lc_arn.tif")
lc_ren = load_labels("data/raw/lc_ren.tif")
print(lc_arn.shape)

# define the labels of the training  and validation set and flatten them 
y_train = flatten_labels(lc_arn)
y_val = flatten_labels(lc_ren)

print("The size of the training set with labes is : ",y_train.shape)
print(np.unique(y_train))   

# train the model 
#model = make_rf()
#model_used = "RF"
model = make_knn()
model_used = "KNN"
model.fit(X_train,y_train)

# Evaluate on Renkum 
y_pred = model.predict(X_val)
class_ids = sorted(np.unique(y_train))
print("Renkum macro IoU: ", macro_iou(y_val,y_pred))
print("Per class IoU: ", per_class_iou(y_val,y_pred, class_ids))

#create an output file 
os.makedirs("outputs", exist_ok=True)

#make the predictions of the model an image again 
height,width = s2_ren.shape[1:]
pred_map = unflatten_predictions(y_pred,height=height, width=width)

# true map vs predicted map for Renkum
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_map(lc_ren, "Renkum: true", axs[0])
plot_map(pred_map, f"Renkum: predicted ({model_used})", axs[1])
plt.tight_layout()
plt.savefig(f"outputs/renkum_map_{model_used}.png", dpi=150)
plt.show()

# legend (separate figure)
plot_legend()
plt.savefig("outputs/legend.png", dpi=150, bbox_inches="tight")
plt.show()

