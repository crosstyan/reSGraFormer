import logging
import os
import random
from pathlib import Path
from typing import Final, Literal, Optional, TypeVar, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from common.h36m_dataset import Human36mDataset
from common.Mydataset import Fusion
from common.opt import Options
from common.utils import (
    CumLoss,
    Split,
    define_error_list,
    get_varialbe,
    mpjpe_cal,
    print_error,
    save_model,
    save_model_epoch,
    test_calculation,
    remove_module_prefix,
)
from model.sgra_former import SGraFormer

TrainModel = Union[SGraFormer, nn.DataParallel[SGraFormer]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_IDs: Final[list[int]] = [0, 1]
T = TypeVar("T")


def train(
    opt: Options,
    actions: list[str],
    train_loader: DataLoader[T],
    model: TrainModel,
    optimizer: Optimizer,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
):
    return step(
        "train",
        opt,
        actions,
        train_loader,
        model,
        optimizer,
        epoch,
        writer,
    )


def val(
    opt: Options,
    actions: list[str],
    val_loader: DataLoader[T],
    model: TrainModel,
):
    with torch.no_grad():
        return step("test", opt, actions, val_loader, model)


def step(
    split: Split,
    opt: Options,
    actions: list[str],
    dataLoader: DataLoader[T],
    model: TrainModel,
    optimizer: Optional[Optimizer] = None,
    epoch: int = 1,
    writer: Optional[SummaryWriter] = None,
):
    loss_all = {"loss": CumLoss()}
    action_error_sum = define_error_list(actions)

    if split == "train":
        model.train()
    else:
        model.eval()

    TQDM = tqdm(enumerate(dataLoader), total=len(dataLoader), ncols=100)
    for i, data in TQDM:
        data = cast(Fusion.GetItemData, data)
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, start, end, hops = (
            data
        )

        [input_2D, gt_3D, _, scale, bb_box, hops] = get_varialbe(
            split, [input_2D, gt_3D, batch_cam, scale, bb_box, hops]
        )

        def input_augmentation(input_2D: Tensor, hops: Tensor, model: nn.Module):
            input_2D_non_flip = input_2D[:, 0]
            output_3D_non_flip = cast(Tensor, model(input_2D_non_flip, hops))

            return input_2D_non_flip, output_3D_non_flip

        if split == "train":
            output_3D = model(input_2D, hops)
        elif split == "test":
            input_2D, output_3D = input_augmentation(input_2D, hops, model)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == "train":
            loss = mpjpe_cal(output_3D, out_target)

            TQDM.set_description(f"Epoch [{epoch}/{opt.nepoch}]")
            TQDM.set_postfix({"l": loss.item()})

            N = input_2D.size(0)
            loss_all["loss"].update(float(loss.detach().cpu().numpy()) * N, N)

            assert (
                optimizer is not None
            ), "train mode requires optimizer but None is given"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer is not None:
                writer.add_scalars(
                    main_tag="scalars1/train_loss",
                    tag_scalar_dict={"trianloss": loss.item()},
                    global_step=(epoch - 1) * len(dataLoader) + i,
                )

        elif split == "test":
            if output_3D.shape[1] != 1:
                output_3D = output_3D[:, opt.pad].unsqueeze(1)
            output_3D[:, :, 1:, :] -= output_3D[:, :, :1, :]
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation(
                output_3D, out_target, action, action_error_sum, opt.dataset, subject
            )

    if split == "train":
        return loss_all["loss"].avg
    elif split == "test":
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2


def main(opt: Options):
    if opt.manual_seed is not None:
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
    root_path = opt.root_path
    if not opt.train and not opt.test:
        raise ValueError("either train or test should be True")

    if opt.train:
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            filename=os.path.join(opt.checkpoint, "train.log"),
            level=logging.INFO,
        )

    root_path = Path(opt.root_path)
    dataset_path = root_path / ("data_3d_" + opt.dataset + ".npz")

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)  # type: ignore

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = DataLoader(
            train_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            pin_memory=True,
        )
    else:
        train_dataloader = None

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = DataLoader[Fusion](
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        pin_memory=True,
    )

    model = SGraFormer(
        num_frame=opt.frames,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.1,
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=CUDA_IDs).to(device)
    else:
        model = model.to(device)

    if opt.previous_dir != "":
        print("pretrained model path:", opt.previous_dir)
        model_path = opt.previous_dir
        pre_dict = torch.load(model_path)
        state_dict = remove_module_prefix(pre_dict)
        model.load_state_dict(state_dict, strict=True)

    all_param = []
    lr = opt.lr
    all_param += list(model.parameters())

    optimizer = AdamW(all_param, lr=lr, weight_decay=0.1)
    for epoch in range(1, opt.nepoch + 1):
        assert train_dataloader is not None, "train_dataloader is None"
        p1, p2 = val(opt, actions, test_dataloader, model)  # type: ignore
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)
        else:
            loss = 0.0

        if opt.train:
            save_model_epoch(Path(opt.checkpoint), epoch, model)

            if p1 < opt.previous_best_threshold:
                opt.previous_name = save_model(opt, epoch, p1, model, "sgra_former")
                opt.previous_best_threshold = float(p1)

        if not opt.train:
            print("p1: %.2f, p2: %.2f" % (p1, p2))
            break
        else:
            logging.info(
                "epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f"
                % (epoch, lr, loss, p1, p2)
            )
            print(
                "e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f"
                % (epoch, lr, loss, p1, p2)
            )

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= opt.lr_decay
                lr *= opt.lr_decay

    print(opt.checkpoint)


if __name__ == "__main__":
    opt = Options()
    main(opt)
