import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from transformers.utils import logging

# from pretrain_encoder import main as pretrain_encoder
from train import main as train
from utils.training_args import Cfg, instantiate_arg_classes, process_config

OmegaConf.register_new_resolver("eval", eval)
logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

cs = ConfigStore.instance()
cs.store(name="config", node=Cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg: Cfg = instantiate_arg_classes(cfg)
    process_config(cfg)

    if cfg.training.pretrain_encoder:
        raise NotImplementedError("Pre-training encoder is currently not implemented.")
        # pretrain_encoder(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
