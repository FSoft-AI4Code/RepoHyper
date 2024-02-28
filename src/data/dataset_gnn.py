import os
import torch
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
from torch_geometric.data import HeteroData
from src.data.transform import SubgraphsData 
import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from joblib import Parallel, delayed


def adj_to_edge_index(adj):
    return adj.nonzero().t().contiguous()

def expand_cls(predict_idx, index_to_name):
    expand_predict_idx = [predict_idx]
    for index, name in index_to_name.items():
        if name.split(".")[-1] in index_to_name[predict_idx]:
            expand_predict_idx.append(index)
    return list(set(expand_predict_idx))

def map_index_to_func_cls_index(index, function_index, cls_index):
    return function_index.index(index) if index in function_index else len(function_index)-1+cls_index.index(index)

class HeRepoBenchData(Dataset):
    def __init__(self, folder_path, transforms=[T.ToUndirected()]) -> None:
        """RepoBench datasets that are stored in the ./data/repobench/repos_graphs_labeled/ folder.
        Args:
                folder_path (str): folder path to the datasets, pickles files.
        """
        super(HeRepoBenchData, self).__init__()
        self.metadata = None
        self.data = []
        if folder_path is not None:
            self.folder_path = folder_path
            self.file_names = os.listdir(folder_path)
            self.file_paths = [os.path.join(folder_path, file_name)
                            for file_name in self.file_names]
            self.transforms = transforms
            self.loading_data()
            self.metadata = self.data[0].metadata()

    def loading_data(self):
        def loading_data_by_file(file_path):
            # try:
                data_by_file = []
                with open(file_path, "rb") as f:
                    samples = pickle.load(f)
                for sample in samples:
                    if (sample["gold_index"] is None) or (len(sample["embeddings"]) == 0) or (len(sample["edge_tensor"]) == 0) or (len(sample["query"]) == 0) or "import_indexes" not in sample.keys():
                        continue
                    # print(sample["context"])
                    if len([idx for idx in sample["import_indexes"] if idx != None]) == 0:
                        continue
                    if len(sample["embeddings"]) > 1500:
                        continue

                    data = SubgraphsData()
                    cls_index = [index for index in sample["index_to_node_type"].keys(
                    ) if sample["index_to_node_type"][index] == 2]
                    data["classes"].x = sample["embeddings"][cls_index]
                    function_index = [index for index in sample["index_to_node_type"].keys(
                    ) if sample["index_to_node_type"][index] == 1]
                    data["functions"].x = sample["embeddings"][function_index]
                    code_index = [index for index in sample["index_to_node_type"].keys(
                    ) if sample["index_to_node_type"][index] == 0]
                    data["code"].x = sample["embeddings"][code_index]

                    adj_cls_owns_functions = sample["type_edges_tensor"][4][cls_index][:, function_index]
                    data["classes", "owns", "functions"].edge_index = adj_to_edge_index(
                        adj_cls_owns_functions)
                    adj_code_owns_functions = sample["type_edges_tensor"][3][code_index][:, function_index]
                    data["code", "owns", "functions"].edge_index = adj_to_edge_index(
                        adj_code_owns_functions)
                    adj_code_owns_classes = sample["type_edges_tensor"][3][code_index][:, cls_index]
                    data["code", "owns", "classes"].edge_index = adj_to_edge_index(
                        adj_code_owns_classes)
                    adj_functions_call_functions = sample["type_edges_tensor"][1][function_index][:, function_index]
                    data["functions", "call", "functions"].edge_index = adj_to_edge_index(
                        adj_functions_call_functions)
                    adj_classes_call_functions = sample["type_edges_tensor"][1][cls_index][:, function_index]
                    data["classes", "call", "functions"].edge_index = adj_to_edge_index(
                        adj_classes_call_functions)
                    adj_classes_call_classes = sample["type_edges_tensor"][1][cls_index][:, cls_index]
                    data["classes", "call", "classes"].edge_index = adj_to_edge_index(
                        adj_classes_call_classes)
                    adj_code_import_functions = sample["type_edges_tensor"][0][code_index][:, function_index]
                    data["code", "import", "functions"].edge_index = adj_to_edge_index(
                        adj_code_import_functions)
                    adj_code_import_classes = sample["type_edges_tensor"][0][code_index][:, cls_index]
                    data["code", "import", "classes"].edge_index = adj_to_edge_index(
                        adj_code_import_classes)
                    adj_code_call_functions = sample["type_edges_tensor"][2][code_index][:, function_index]
                    data["code", "call", "functions"].edge_index = adj_to_edge_index(
                        adj_code_call_functions)
                    adj_code_call_classes = sample["type_edges_tensor"][2][code_index][:, cls_index]
                    data["code", "call", "classes"].edge_index = adj_to_edge_index(
                        adj_code_call_classes)
                    data = data.to_homogeneous()
                    if self.transforms is not None:
                        for transform in self.transforms:
                            data = transform(data)
                    
                    data["query"] = sample["query"]
                    data["import_indexes"] = [map_index_to_func_cls_index(idx, function_index, cls_index) for idx in sample["import_indexes"]]
                    data["y"] = map_index_to_func_cls_index(sample["gold_index"], function_index, cls_index)
                    data["map_indexes"] = {"classes": cls_index, "functions": function_index, "code": code_index}
                    
                    data_by_file.append(data)
            # except:
            #     data_by_file = []
                return data_by_file
        
        print("Loading data...")
        lst_data = Parallel(n_jobs=1)(delayed(loading_data_by_file)(file_path) for file_path in tqdm(self.file_paths[:20]))
        self.data = [data for lst in lst_data for data in lst]
        print(self.data)

    def len(self):
        return len(self.data)

    def get(self, index):
        return self.data[index]
    
    def split(self, ratio):
        trainset = HeRepoBenchData.from_data_list(self.data[:int(len(self.data)*ratio)])
        valset = HeRepoBenchData.from_data_list(self.data[int(len(self.data)*ratio):])
        return trainset, valset
    
    @classmethod
    def from_data_list(cls, data_list):
        dataset = cls(folder_path=None)
        dataset.data = data_list
        dataset.metadata = data_list[0].metadata()
        return dataset


class RepoBenchData(Dataset):
    def __init__(self, folder_path, transforms=None) -> None:
        """RepoBench datasets that are stored in the ./data/repobench/repos_graphs_labeled/ folder.
                        Args:
                                folder_path (str): folder path to the datasets, pickles files.
        """
        super(RepoBenchData, self).__init__()    
        self.metadata = None
        self.data = []
        if folder_path is not None:
            self.folder_path = folder_path
            self.file_names = os.listdir(folder_path)
            self.file_paths = [os.path.join(folder_path, file_name) for file_name in self.file_names]
            self.loading_data()
            self.transforms = transforms
        
    def loading_data(self):
        def loading_data_by_file(file_path):
            data_by_file = []
            with open(file_path, "rb") as f:
                samples = pickle.load(f)
            for sample in samples:
                if (sample["gold_index"] is None) or (len(sample["embeddings"]) == 0) or (len(sample["edge_tensor"]) == 0) or (len(sample["query"]) == 0) or "import_indexes" not in sample.keys():
                    continue
                # print(sample["context"])
                if len([idx for idx in sample["import_indexes"] if idx != None]) == 0:
                    continue
                if len(sample["embeddings"]) > 1500:
                    continue
                
                x = sample["embeddings"]
                edge_index = adj_to_edge_index(sample["edge_tensor"])
                file_indexes = [index for index in sample["index_to_name"].keys() if sample["index_to_name"][index].endswith(".py")]
                file_indexes = torch.tensor(file_indexes).reshape(1, -1)
                data = Data(x=x, edge_index=edge_index, y=sample["gold_index"], query=sample["query"], 
                            import_indexes=sample["import_indexes"], index_to_name=[expand_cls(idx, sample["index_to_name"]) for 
                                idx in range(len(sample["index_to_name"]))],
                            ignore_index=file_indexes)
                if self.transforms is not None:
                    for transform in self.transforms:
                        data = transform(data)
                # data = T.ToUndirected()(data)
                data_by_file.append(data)
            return data_by_file
        
        print("Loading data...")
        lst_data = Parallel(n_jobs=24)(delayed(loading_data_by_file)(file_path) for file_path in tqdm(self.file_paths))
        self.data = [data for lst in lst_data for data in lst]

    def len(self):
        return len(self.data)

    def get(self, index):
        return self.data[index]
    
    def split(self, ratio):
        trainset = RepoBenchData.from_data_list(self.data[:int(len(self.data)*ratio)])
        valset = RepoBenchData.from_data_list(self.data[int(len(self.data)*ratio):])
        return trainset, valset
    
    @classmethod
    def from_data_list(cls, data_list):
        dataset = cls(folder_path=None)
        dataset.data = data_list
        return dataset
    
    

if __name__ == "__main__":
    RepobenchData = HeRepoBenchData(
        "data/repobench/repos_graphs_labeled_cosine_radius_unix_link")
