"""
No@
Jan 2025
"""
from mmselfsup.registry import DATASETS
from mmseg.datasets.ai4arctic_patches import AI4ArcticPatches as MmsegAI4ArcticPatches

# Register the dataset from MMSegmentation into MMSelfSup registry
@DATASETS.register_module()
class AI4ArcticPatches(MmsegAI4ArcticPatches):
    pass