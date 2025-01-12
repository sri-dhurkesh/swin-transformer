image_size = (224, 224)
patch_size = (4, 4)
in_channels = 3
num_classes = 5
embed_dim = 96
depths = [2, 2, 6, 2]
num_heads = [3, 6, 12, 24]
window_size = (7, 7)
mlp_ratio = (4,)
drop_path_rate = 0.5


# dataset path
dataset = dict(
    batch_size=3,
)
