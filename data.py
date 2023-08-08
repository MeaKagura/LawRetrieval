import json
from torch.utils.data import Dataset, DataLoader

# 文件路径
file_path = "./test.json"

# 打开JSON文件并加载数据
with open(file_path, "r") as json_file:
    test_data = json.load(json_file)


class QueryLaw(Dataset):
    def __init__(self, file_path):
        super().__init__()
        with open(file_path, "r") as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
