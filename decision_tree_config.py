# !!! Attention !!! Use config wisely. In decision_tree file ordinary decision tree without new parameters
# is stored. Do not set combinations like name: decision_tree with new_arguments: True because size of
# inputs might be different so program will fail. Set load_first to false and be calm.

__config_values = {
    "name": "decision_tree",
    "load_first": False,
    "shorten": False,
    "normalize": True,
    "new_arguments": False,
    "save": False,
    "cross_validate": True
}