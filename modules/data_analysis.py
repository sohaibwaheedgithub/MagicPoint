import numpy as np
from glob import glob
from tqdm import tqdm


# To check the range of coordinates
pointsFilePaths = glob("dataset/data/*/points/*/*")

widthLowerBound = np.inf
heightLowerBound = np.inf

widthUpperBound = -np.inf
heightUpperBound = -np.inf


for fp in tqdm(pointsFilePaths):
    points = np.load(fp)
    
    try:    
        heightLowerBound = np.minimum(np.min(points[:, 0]), heightLowerBound)
        heightUpperBound = np.maximum(np.max(points[:, 0]), heightUpperBound)
        
        widthLowerBound = np.minimum(np.min(points[:, 1]), widthLowerBound)
        widthUpperBound = np.maximum(np.max(points[:, 1]), widthUpperBound)
    except:
        pass
    
    
print(f"Height Range: ({heightLowerBound}, {heightUpperBound})")
print(f"Width Range: ({widthLowerBound}, {widthUpperBound})")