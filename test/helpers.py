import argparse


def run_model_tests_for(args, main_method):
    test_vit_cifar10(args, main_method)
    test_vit_fashion_mnist(args, main_method)
    test_convnext_cifar10(args, main_method)
    test_convnext_fashion_mnist(args, main_method)
    test_mobilevit_cifar10(args, main_method)
    test_mobilevit_fashion_mnist(args, main_method)
    test_convnext_fiftyone(args, main_method)


def test_vit_cifar10(args, main):
    args.dataset = 'cifar10'
    args.model = 'vit_tiny'
    args.input_channels = 3
    main(args)


def test_vit_fashion_mnist(args, main):
    args.dataset = 'fashion-mnist'
    args.model = 'vit_tiny'
    args.input_channels = 1
    main(args)


def test_convnext_cifar10(args, main):
    args.dataset = 'cifar10'
    args.model = 'convnext_pico'
    args.input_channels = 3
    main(args)


def test_convnext_fashion_mnist(args, main):
    args.dataset = 'fashion-mnist'
    args.model = 'convnext_pico'
    args.input_channels = 1
    main(args)


def test_mobilevit_cifar10(args, main):
    args.dataset = 'cifar10'
    args.model = 'mobilevitv2_050'
    args.input_channels = 3
    main(args)


def test_mobilevit_fashion_mnist(args, main):
    args.dataset = 'fashion-mnist'
    args.model = 'mobilevitv2_050'
    args.input_channels = 1
    main(args)


def test_convnext_fiftyone(args, main):
    args.dataset = 'fiftyone'
    args.model = 'convnext_pico'
    args.input_channels = 1
    main(args)


def get_base_test_args():
    return argparse.Namespace(
        batch_size=17,
        ckpt_path=None,
        clip_grad=3.0,
        compare=None,
        train_subset=34,
        test_subset=34,
        device='cpu',
        epochs=1,
        eval=True,
        freeze_last_layer=1,
        global_crops_scale=(0.7, 1.0),
        local_crop_input_factor=1,
        norm_last_layer=False,
        input_channels=3,
        input_size=32,
        learning_rate=0.0001,
        local_crops_scale=(0.2, 0.5),
        lr_decay=0.5,
        min_lr=1e-06,
        momentum=0.9,
        momentum_teacher=0.9995,
        n_local_crops=8,
        num_classes=0,
        num_workers=1,
        optimizer='adamw',
        method='simclr',
        out_dim=1024,
        patch_size=4,
        resume=False,
        sam=False,
        scheduler=None,
        seed=420,
        teacher_temp=0.04,
        visualize='dino_attn',
        wandb=False,
        warmup_epochs=0,
        warmup_steps=3,
        warmup_teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        weight_decay=0.04,
        weight_decay_end=0.4
    )
