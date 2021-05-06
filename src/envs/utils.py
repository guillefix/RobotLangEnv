from object2urdf import ObjectUrdfBuilder


def generate_urdfs(object_folder="./shapenet_objects/"):
    # Build entire libraries of URDFs
    builder = ObjectUrdfBuilder(object_folder)
    builder.build_library(force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'top')


if __name__ == '__main__':
    import os
    path = os.path.dirname(__file__)
    generate_urdfs(path + "/shapenet_objects/" )