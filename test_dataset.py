import os

data_dir = "dataset"

for category in ["normal", "stroke"]:
    path = os.path.join(data_dir, category)
    print(category, "images:", len(os.listdir(path)))
