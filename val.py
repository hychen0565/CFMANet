"""独立评估入口。

评估逻辑统一放在 train.py（build_val_loader / validate），这里只解析配置、
构建模型与验证集 DataLoader，然后调用 validate()。

用法：
    python val.py --config config/fold0_resnet50.yaml --resume train_epoch_xx_xxx.pth
其中 --resume 是 snapshot 目录下的权重文件名，未指定时使用 YAML 的 TRAIN.resume。
"""
import argparse

import train as trainer
from util import config
from util.util import get_logger


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation (evaluation)')
    parser.add_argument('--config', type=str, default='config/fold0_resnet50.yaml', help='config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint filename inside the YAML snapshot_path (overrides TRAIN.resume)')
    cli_args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli_args.config)
    if cli_args.resume is not None:
        cfg.resume = cli_args.resume
    cfg.evaluate = True
    return cfg


def main():
    trainer.args = get_parser()
    trainer.logger = get_logger()
    if not trainer.args.resume:
        raise ValueError('No checkpoint given: pass --resume <filename> or set TRAIN.resume in the YAML config')

    model, _ = trainer.get_model(trainer.args)
    val_loader = trainer.build_val_loader(trainer.args)
    trainer.validate(val_loader, model)


if __name__ == '__main__':
    main()
