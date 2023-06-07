from models.models import get_eval_model
from train_utils import get_data_loaders, compute_knn
from utils import get_args, fix_seeds
from visualize import is_timm_compatible


def main(args):
    print(f'Running kNN eval for {args.model} model...')

    fix_seeds(args.seed)

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

    _, train_loader_plain, val_loader_plain, _ = get_data_loaders(args)
    knn_acc, precision, recall, f1 = compute_knn(model, train_loader_plain, val_loader_plain)
    print('------------------------')
    print(f'kNN accuracy: \t\t {knn_acc}')
    print(f'Precision: \t\t {precision}')
    print(f'Recall: \t\t {recall}')
    print(f'F1: \t\t {f1}')


if __name__ == '__main__':
    main(get_args())
