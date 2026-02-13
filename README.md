# project-module-cv
Computer Vision approach for PM

1. Generate synthetic data - generate_floorplans.py
2. Tran Unet model trai_unet.py
3. Extract Graph extract.py


### Command to generate synthetic floor plans
python generate_floorplans.py --n_images 1000 --min_rooms 6 --max_rooms 14 --outdir out/
