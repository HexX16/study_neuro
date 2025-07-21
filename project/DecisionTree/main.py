import math
from collections import Counter
from DesTree import DesTree

def print_tree(node, indent=""):
    if node.label is not None:
        print(indent + "Label:", node.label)
    else:
        print(indent + f"{node.feature} <= {node.threshold}")
        print(indent + "├── Yes:")
        print_tree(node.left, indent + "│   ")
        print(indent + "└── No:")
        print_tree(node.right, indent + "    ") 



data = [
    {"power": 150, "weight": 1000, "Cd": 0.30, "label": 0},  # лёгкая, слабая, обычная
    {"power": 160, "weight": 1100, "Cd": 0.32, "label": 0},
    {"power": 170, "weight": 1300, "Cd": 0.31, "label": 0},
    {"power": 200, "weight": 1400, "Cd": 0.29, "label": 1},  # средняя
    {"power": 210, "weight": 1450, "Cd": 0.28, "label": 1},
    {"power": 220, "weight": 1500, "Cd": 0.34, "label": 1},
    {"power": 250, "weight": 1550, "Cd": 0.27, "label": 1},  # быстрая
    {"power": 230, "weight": 1700, "Cd": 0.36, "label": 0},  # тяжелая
    {"power": 240, "weight": 1800, "Cd": 0.35, "label": 0},
    {"power": 260, "weight": 1850, "Cd": 0.33, "label": 0},
]

row = {"power": 260, "weight": 1500, "Cd": 0.29}

tree = DesTree()
tree.tree = tree.build_tree(data)
print_tree(tree.tree)
print("Prediction:", tree.predict_one(row))