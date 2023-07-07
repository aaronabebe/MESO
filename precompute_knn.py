import numpy as np
import os

from data import get_dataloader
from fo_utils import get_dataset
from models.models import get_eval_model
from utils import get_args, fix_seeds_set_flags, compute_embeddings, get_knn_cache_path
from visualize import is_timm_compatible



def main(args):
    print(f'=> Precomputing kNN embeddings for {args.model} and dataset {args.dataset}...')

    fix_seeds_set_flags(args.seed)

    save_path = get_knn_cache_path(args)

    model = get_eval_model(
        args.model,
        args.device,
        args.dataset,
        path_override=args.ckpt_path,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if is_timm_compatible(args.model) else None,
        img_size=args.input_size if is_timm_compatible(args.model) else None,
        load_remote=args.wandb,
        pretrained=args.timm,
        _remove_head=args.ckpt_path
    )

    fo_dataset = None
    if args.dataset == 'fiftyone':
        fo_dataset, _ = get_dataset(
            dataset_dir=args.fo_dataset_dir,
            min_crop_size=1
        )
    dl = get_dataloader(
        args.dataset,
        fo_dataset=fo_dataset,
        train=False,
        subset=args.test_subset,
        batch_size=args.batch_size
    )
    embs, labels = compute_embeddings(args, model, dl, is_training=True)
    np.savez_compressed(save_path, embs=embs, labels=labels)


if __name__ == '__main__':
    main(get_args())
