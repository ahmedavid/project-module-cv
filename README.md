# project-module-cv
Computer Vision approach for PM

1. Generate synthetic data - generate_floorplans.py
2. Tran Unet model trai_unet.py
3. Extract Graph extract.py


### Command to generate synthetic floor plans
python generate_floorplans.py --n_images 1000 --min_rooms 6 --max_rooms 14 --outdir out/


## Performance Metrics
The training script evaluates model quality using two metrics computed on the validation set after each epoch.

###Pixel Accuracy
Pixel accuracy is the simplest segmentation metric. It measures the percentage of pixels that were classified correctly:


pixel_accuracy = correct_pixels / total_pixels
A prediction is compared element-wise against the ground-truth mask. While easy to interpret, pixel accuracy can be misleading when classes are imbalanced — a model that predicts only the background class may still achieve high pixel accuracy if background dominates the image.

### Mean Intersection-over-Union (mIoU)
mIoU is the standard metric for semantic segmentation. For each class, it computes the overlap between the predicted and ground-truth regions:


IoU(class) = intersection(pred, target) / union(pred, target)
The mean is then taken across all classes that are actually present in the ground truth. This avoids inflating the score with trivially correct classes that have zero pixels in both prediction and target.

mIoU penalizes both false positives and false negatives equally, making it a more reliable indicator of segmentation quality than pixel accuracy alone.
