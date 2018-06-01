from envs import FourbarDataset
from train import parseArgs, params

dataset = FourbarDataset(compare_all_branches=False)

dataset.compute(num=1000)

dataset.save()


