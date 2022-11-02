def get_experiment_name(args):
    name = f'{args.model}_{args.epochs}_{args.batch_size}'
    return name
