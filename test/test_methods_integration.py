from test.helpers import get_base_test_args, run_model_tests_for
from train_contrastive import main as main_contrastive
from train_linear import main as main_linear
from train_ssl import main as main_dino


def test_supcon():
    args = get_base_test_args()
    args.method = 'supcon'
    run_model_tests_for(args, main_contrastive)


def test_simclr():
    args = get_base_test_args()
    args.method = 'simclr'
    run_model_tests_for(args, main_contrastive)


def test_dino():
    args = get_base_test_args()
    run_model_tests_for(args, main_dino)


def test_linear():
    args = get_base_test_args()
    run_model_tests_for(args, main_linear)
