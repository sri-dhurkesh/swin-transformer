"""Main file to run a training"""

from torchvision.datasets import ImageFolder
from torch.utils.data import dataloader, random_split
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, random_split

# from mmpretrain.evaluation.metrics import Accuracy
from mmengine import Config
from torch.optim import SGD
from mmengine.runner import Runner
from models.swin_v1.swin_transformer import SwinTransformer


cfg = Config.fromfile("/workspaces/swin-transformer/configs/swin_v1_config.py")
dataset_cfg = cfg.dataset

# Define the train_transform using Compose
train_transform = Compose([Resize((224, 224)), ToTensor()])

# Define the test_transform using Compose
test_transform = Compose([Resize((224, 224)), ToTensor()])

# create a training dataset
train_data = ImageFolder("data/logos-bk-kfc-mcdonald-starbucks-subway-none/logos3/train", transform=train_transform)
train_count = int(len(train_data) * 0.80)
val_count = len(train_data) - train_count
train_set, val_set = random_split(train_data, [train_count, val_count])

# dataloader
train_dataloader = dataloader.DataLoader(train_set, batch_size=dataset_cfg.batch_size, shuffle=False)
val_dataloader = dataloader.DataLoader(val_set, batch_size=dataset_cfg.batch_size, shuffle=False)

# creating a test dataloader
test_data = ImageFolder("data/logos-bk-kfc-mcdonald-starbucks-subway-none/logos3/test", transform=test_transform)
test_dataloader = dataloader.DataLoader(test_data, batch_size=dataset_cfg.batch_size, shuffle=False)
from mmengine.evaluator import BaseMetric


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # save the middle result of a batch to `self.results`
        self.results.append(
            {
                "batch_size": len(gt),
                "correct": (score.argmax(dim=1) == gt).sum().cpu(),
            }
        )

    def compute_metrics(self, results):
        total_correct = sum(item["correct"] for item in results)
        total_size = sum(item["batch_size"] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict(accuracy=100 * total_correct / total_size)


# runner
runner = Runner(
    # the model used for training and validation.
    # Needs to meet specific interface requirements
    model=SwinTransformer(**cfg.to_dict()),
    # working directory which saves training logs and weight files
    work_dir="./work_dir",
    # train dataloader needs to meet the PyTorch data loader protocol
    train_dataloader=train_dataloader,
    # optimize wrapper for optimization with additional features like
    # AMP, gradtient accumulation, etc
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # trainging coinfs for specifying training epoches, verification intervals, etc
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # validation dataloaer also needs to meet the PyTorch data loader protocol
    val_dataloader=val_dataloader,
    # validation configs for specifying additional parameters required for validation
    val_cfg=dict(),
    # validation evaluator. The default one is used here
    val_evaluator=dict(type=Accuracy),
)
runner.train()
