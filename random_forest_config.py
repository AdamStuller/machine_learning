# !!! Attention !!! Use config wisely. In decision_tree file ordinary decision tree without new parameters
# is stored. Do not set combinations like name: random_forest with new_arguments: True because size of
# inputs might be different so program will fail. Set load_first to false and be calm.

__config_values = {
    "name": "random_forest",
    "load_first": False,
    "shorten": True,
    "normalize": True,
    "new_arguments": True,
    "save": False,
    "cross_validate": True
}