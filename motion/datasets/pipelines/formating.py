from ..registry import PIPELINES


@PIPELINES.register_module
class Reformat(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        data_bundle = dict(
            metadata=res["metadata"])

        data_bundle.update(res["lidar"])
        data_bundle.update(res["voxel"])

        return data_bundle, info
