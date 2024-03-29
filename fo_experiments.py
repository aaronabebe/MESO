import argparse

import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import torch
from fiftyone import ViewField as F
from torch.utils.data import Subset
from tqdm import tqdm

from data import SailingLargestCropDataset, default_empty_transforms
from fo_utils import GROUND_TRUTH_LABEL, get_dataset, DATASET_NAME, DATASET_DIR

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
    mean, std, nb_samples = 0., 0., 0.

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


def main(args):
    fo_dataset, _ = get_dataset(dataset_dir=args.fo_dataset_dir, use_16bit=args.use_16bit)
    torch_dataset = SailingLargestCropDataset(fo_dataset, transform=default_empty_transforms(),
                                              use_16bit=args.use_16bit)
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


def server_main(args):
    # dataset, _ = get_dataset(split=(1.0, 0.0), use_16bit=args.use_16bit)
    fo.config.module_path = "custom_embedded_files"
    dataset = fo.Dataset.from_dir(
        dataset_dir=DATASET_DIR,
        dataset_type=fo.types.FiftyOneDataset,
        name=DATASET_NAME,
        label_field=GROUND_TRUTH_LABEL
    )
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    # TODO set envs via code and not via shell
    # export PYTHONPATH="${PYTHONPATH}:/home/aaron/thesis/repos/SSL_MFE/"
    # export FIFTYONE_MODULE_PATH=custom_embedded_files
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch_server", action='store_true', default=False)
    parser.add_argument("--mean_std", action='store_true', default=False)
    parser.add_argument("--use_16bit", action='store_true', default=False)
    args = parser.parse_args()

    if args.launch_server:
        server_main(args)
    else:
        main(args)
