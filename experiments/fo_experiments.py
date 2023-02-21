import argparse

import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import torch
from fiftyone import ViewField as F
from torch.utils.data import Subset
from tqdm import tqdm
from torchvision import transforms

from data import FiftyOneTorchDataset, default_fifty_one_transforms
from fo_utils import DATASET_NAME, GROUND_TRUTH_LABEL, DATASET_DIR, get_dataset

MIN_CROP_SIZE = 32


def get_classes_dict(dataset: fo.Dataset):
    class_view = dataset.match(F(f"{GROUND_TRUTH_LABEL}.detections.label") > 0)

    class_count = {}

    for s in class_view:
        for d in s.ground_truth_det.detections:
            if d.label not in class_count:
                class_count[d.label] = 1
            else:
                class_count[d.label] += 1
    return class_count


def calc_mean_std(loader: torch.utils.data.DataLoader):
    channels_sum, channels_square_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        data = data / 255
        channels_sum += torch.mean(data, dim=[])
        channels_square_sum += torch.mean(data ** 2, dim=[])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_square_sum / num_batches - mean ** 2)

    return mean, std


def main(args):
    fo_dataset, _ = get_dataset()
    torch_dataset = FiftyOneTorchDataset(fo_dataset, transform=default_fifty_one_transforms())
    print('LEN DATASET TOTAL: ', len(torch_dataset))
    if args.mean_std:
        mean, std = calc_mean_std(torch.utils.data.DataLoader(torch_dataset, batch_size=1, shuffle=False))
        print('MEAN: ', mean)
        print('STD: ', std)
        return

    # TODO move this to visualize.py
    subset = Subset(torch_dataset, range(16))
    loader = torch.utils.data.DataLoader(subset, shuffle=True, num_workers=0)

    cropped_images = [img for img, label in loader]

    n = int(np.ceil(len(cropped_images) ** .5))
    fig, axs = plt.subplots(n, n, figsize=(n * 3, n * 3))
    for i, img in enumerate(cropped_images):
        ax = axs[i // n][i % n]
        ax.imshow(img.squeeze().numpy())
        ax.axis("off")
    fig.tight_layout()
    plt.savefig('test.png')
    plt.show()


def server_main(dataset_name: str = DATASET_NAME, dataset_dir: str = DATASET_DIR,
                ground_truth_label: str = GROUND_TRUTH_LABEL):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name,
        label_field=ground_truth_label
    )

    session = fo.launch_app(dataset)
    session.wait()

    return dataset


if __name__ == '__main__':
    # TODO set envs via code and not via shell
    # export PYTHONPATH="${PYTHONPATH}:/home/aaron/thesis/repos/SSL_MFE/"
    # export FIFTYONE_MODULE_PATH=custom_embedded_files
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch_server", action='store_true', default=False)
    parser.add_argument("--mean_std", action='store_true', default=False)
    args = parser.parse_args()
    if args.launch_server:
        server_main()
    else:
        main(args)
