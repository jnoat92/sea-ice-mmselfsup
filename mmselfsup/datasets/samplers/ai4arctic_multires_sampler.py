from mmselfsup.registry import DATA_SAMPLERS
from mmseg.structures.sampler.ai4arctic_multires_sampler import WeightedInfiniteSampler as MmsegWeightedInfiniteSampler

# Register the dataset from MMSegmentation into MMSelfSup registry
@DATA_SAMPLERS.register_module()
class WeightedInfiniteSampler(MmsegWeightedInfiniteSampler):
    pass