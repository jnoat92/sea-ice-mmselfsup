"""
No@
Jan 2025
"""
from mmselfsup.registry import TRANSFORMS
from mmseg.datasets.transforms.loading_ai4arctic_patches import LoadPatchFromPKLFile as MmsegLoadPatchFromPKLFile

# Register the dataset from MMSegmentation into MMSelfSup registry
@TRANSFORMS.register_module()
class LoadPatchFromPKLFile(MmsegLoadPatchFromPKLFile):
    pass