import kenn as knn
import torch
import numpy as np


preactivations = (torch.rand(20, 2) - 0.5) * 100
indexing_training = torch.Tensor([[1, 3], [4, 6], [7, 5], [11, 4], [12, 13], [15, 17], [18, 19]])
relations = torch.ones(7, 1) * 500

print(torch.squeeze(preactivations))
kenn_layer = knn.relational_parser("logic_file.txt")
kenn_layer(preactivations,
           relations,
           indexing_training[:, 0].long(),
           indexing_training[:, 1].long())
