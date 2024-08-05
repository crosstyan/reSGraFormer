import logging
import os
import random
from pathlib import Path
from typing import Final, Literal, Optional, TypeVar, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils as utils
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
from common.utils import (AccumLoss, Split, define_error_list, get_varialbe,
                          mpjpe_cal, print_error, save_model, save_model_epoch,
                          test_calculation)
from model.sgra_former import SGraFormer

TrainModel = Union[SGraFormer, nn.DataParallel[SGraFormer]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_IDs: Final[list[int]] = [0, 1]
T = TypeVar("T")


def visualize_skeletons(input_2D, output_3D, gt_3D, idx=0, output_dir="./output"):
    # Ensure the tensors are on the CPU and convert them to numpy arrays
    input_2D = input_2D.cpu().numpy()
    output_3D = output_3D.cpu().numpy()
    gt_3D = gt_3D.cpu().numpy()

    # print("====> input_2D: ", input_2D[-1])
    # Get the first action and first sample from the batch
    input_sample = input_2D[idx, 0]
    output_sample = output_3D[idx, 0]
    gt_3D_sample = gt_3D[idx, 0]

    print(f"\ninput_sample shape: {input_sample.shape}")
    print(f"output_sample shape: {output_sample.shape}")

    fig = plt.figure(figsize=(25, 5))

    # Define the connections (bones) between joints
    bones = [
        (0, 1),
        (1, 2),
        (2, 3),  # Left leg
        (0, 4),
        (4, 5),
        (5, 6),  # Right leg
        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),  # Spine
        (7, 11),
        (11, 12),
        (12, 13),  # Right arm
        (7, 14),
        (14, 15),
        (15, 16),  # Left arm
    ]

    # Colors for different parts
    bone_colors = {"leg": "green", "spine": "blue", "arm": "red"}

    # Function to get bone color based on index
    def get_bone_color(start, end):
        if (
            start in [1, 2, 3]
            or end in [1, 2, 3]
            or start in [4, 5, 6]
            or end in [4, 5, 6]
        ):
            return bone_colors["leg"]
        elif start in [7, 8, 9, 10] or end in [7, 8, 9, 10]:
            return bone_colors["spine"]
        else:
            return bone_colors["arm"]

    # Plotting 2D skeletons from different angles
    for i in range(4):
        ax = fig.add_subplot(1, 7, i + 1)
        ax.set_title(f"2D angle {i+1}")
        ax.scatter(input_sample[i, :, 0], input_sample[i, :, 1], color="blue")

        # Draw the bones
        for start, end in bones:
            bone_color = get_bone_color(start, end)
            ax.plot(
                [input_sample[i, start, 0], input_sample[i, end, 0]],
                [input_sample[i, start, 1], input_sample[i, end, 1]],
                color=bone_color,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(
            np.min(input_sample[:, :, 0]) - 1, np.max(input_sample[:, :, 0]) + 1
        )
        ax.set_ylim(
            np.min(input_sample[:, :, 1]) - 1, np.max(input_sample[:, :, 1]) + 1
        )
        ax.grid()

    # Plotting predicted 3D skeleton
    ax = fig.add_subplot(1, 7, 5, projection="3d")
    ax.set_title("3D Predicted Skeleton")
    ax.scatter(
        output_sample[:, 0],
        output_sample[:, 1],
        output_sample[:, 2],
        color="red",
        label="Predicted",
    )

    # Draw the bones in 3D for output_sample
    for start, end in bones:
        bone_color = get_bone_color(start, end)
        ax.plot(
            [output_sample[start, 0], output_sample[end, 0]],
            [output_sample[start, 1], output_sample[end, 1]],
            [output_sample[start, 2], output_sample[end, 2]],
            color=bone_color,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.set_xlim(np.min(output_sample[:, 0]) - 1, np.max(output_sample[:, 0]) + 1)
    ax.set_ylim(np.min(output_sample[:, 1]) - 1, np.max(output_sample[:, 1]) + 1)
    ax.set_zlim(np.min(output_sample[:, 2]) - 1, np.max(output_sample[:, 2]) + 1)  # type: ignore
    ax.legend()

    # Plotting ground truth 3D skeleton
    ax = fig.add_subplot(1, 7, 6, projection="3d")
    ax.set_title("3D Ground Truth Skeleton")
    ax.scatter(
        gt_3D_sample[:, 0],
        gt_3D_sample[:, 1],
        gt_3D_sample[:, 2],
        color="blue",
        label="Ground Truth",
    )

    # Draw the bones in 3D for gt_3D_sample
    for start, end in bones:
        bone_color = get_bone_color(start, end)
        ax.plot(
            [gt_3D_sample[start, 0], gt_3D_sample[end, 0]],
            [gt_3D_sample[start, 1], gt_3D_sample[end, 1]],
            [gt_3D_sample[start, 2], gt_3D_sample[end, 2]],
            color=bone_color,
            linestyle="--",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.set_xlim(np.min(gt_3D_sample[:, 0]) - 1, np.max(gt_3D_sample[:, 0]) + 1)
    ax.set_ylim(np.min(gt_3D_sample[:, 1]) - 1, np.max(gt_3D_sample[:, 1]) + 1)
    ax.set_zlim(np.min(gt_3D_sample[:, 2]) - 1, np.max(gt_3D_sample[:, 2]) + 1)  # type: ignore
    ax.legend()

    plt.grid()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/skeletons_visualization.png")
    plt.show()


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
    loss_all = {"loss": AccumLoss()}
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

        if split == "train":
            output_3D = model(input_2D, hops)
        elif split == "test":
            input_2D, output_3D = input_augmentation(input_2D, hops, model)

        visualize_skeletons(input_2D, output_3D, gt_3D)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == "train":
            loss = mpjpe_cal(output_3D, out_target)

            TQDM.set_description(f"Epoch [{epoch}/{opt.nepoch}]")
            TQDM.set_postfix({"l": loss.item()})

            N = input_2D.size(0)
            loss_all["loss"].update(loss.detach().cpu().numpy() * N, N)

            assert optimizer is not None, "train mode requires optimizer but None"
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


def input_augmentation(input_2D, hops, model):
    input_2D_non_flip = input_2D[:, 0]
    output_3D_non_flip = model(input_2D_non_flip, hops)

    return input_2D_non_flip, output_3D_non_flip


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

    def remove_module_prefix(
        state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        return new_state_dict

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
        print("=====> p1, p2", p1, p2)
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)
        else:
            loss = 0.0

        if opt.train:
            save_model_epoch(opt.checkpoint, epoch, model)

            if p1 < opt.previous_best_threshold:
                opt.previous_name = save_model(
                    opt.previous_name, opt.checkpoint, epoch, p1, model
                )
                opt.previous_best_threshold = p1

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
