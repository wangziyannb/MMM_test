import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MOTIONGPT3_ROOT = REPO_ROOT / "third_party" / "MotionGPT3"

STAGE_CONFIGS = {
    "stage1_t2m": "configs/MoT_vae_stage1_t2m.yaml",
    "stage2_all": "configs/MoT_vae_stage2_all.yaml",
    "stage2_instruct": "configs/MoT_vae_stage2_instruct.yaml",
    "stage3_finetune": "configs/MoT_vae_stage3.yaml",
}

STAGE_TASK_FILES = {
    "stage1_t2m": "prepare/instructions/template_t2m_pretrain.json",
    "stage2_all": "prepare/instructions/template_pretrain.json",
    "stage2_instruct": "prepare/instructions/template_instructions.json",
    "stage3_finetune": "prepare/instructions/template_witht2t_instructions.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train official MotionGPT3 and evaluate it with this repo's "
            "motion/text evaluation code."
        )
    )
    parser.add_argument("--motiongpt3-root", default=os.environ.get("MOTIONGPT3_ROOT", str(DEFAULT_MOTIONGPT3_ROOT)))
    parser.add_argument("--mgpt3-stage", choices=sorted(STAGE_CONFIGS), default="stage2_all")
    parser.add_argument("--cfg", default=None, help="Official MotionGPT3 config path, relative to --motiongpt3-root or absolute.")
    parser.add_argument("--cfg-assets", default="configs/assets.yaml", help="Official MotionGPT3 assets config.")
    parser.add_argument("--nodebug", action="store_true", help="Match official --nodebug.")

    parser.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "motiongpt3"))
    parser.add_argument("--exp-name", default="motiongpt3_stage2_all_our_eval")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset" / "HumanML3D"))
    parser.add_argument("--glove-root", default=str(REPO_ROOT / "glove"))
    parser.add_argument("--eval-opt", default=str(REPO_ROOT / "checkpoints" / "t2m" / "Comp_v6_KLD005" / "opt.txt"))
    parser.add_argument("--eval-dataname", choices=["t2m", "kit"], default="t2m")
    parser.add_argument("--task-path", default=None, help="Instruction json for official MotionGPT3 training.")

    parser.add_argument("--pretrained-vae", default=None, help="Official MotionGPT3/MotionVAE checkpoint.")
    parser.add_argument("--pretrained", default=None, help="Official MotionGPT3 checkpoint to load before training.")
    parser.add_argument("--resume", default=None, help="Official MotionGPT3 experiment folder to resume.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load an official MotionGPT3 checkpoint and run only this repo's evaluation.")
    parser.add_argument("--checkpoint", default=None,
                        help="Official MotionGPT3 checkpoint for --eval-only. Relative paths are resolved from this repo.")
    parser.add_argument("--eval-split", default=None,
                        help="Override cfg.EVAL.SPLIT for --eval-only, e.g. val or test.")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--end-epoch", type=int, default=None, help="Official TRAIN.END_EPOCH.")
    parser.add_argument("--eval-every-epoch", type=int, default=None, help="Run our eval every N epochs.")
    parser.add_argument("--eval-every-step", type=int, default=None,
                        help="Run our eval every N optimizer steps. Old --eval-iter maps here.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=int, nargs="+", default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--replication-times", type=int, default=1)
    parser.add_argument("--t2m-repeats", type=int, default=1)
    parser.add_argument("--m2t-repeats", type=int, default=1)
    parser.add_argument("--skip-our-eval", action="store_true")
    parser.add_argument("--keep-official-val", action="store_true",
                        help="Keep official validation loop in addition to our eval callback.")
    parser.add_argument("--keep-official-metrics", action="store_true",
                        help="Do not clear official METRIC.TYPE. Useful only if official evaluator deps are installed.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable official wandb logger.")

    # Accepted only to make old train_bitm commands fail softly.
    parser.add_argument("--vq-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--block-size", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--total-iter", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-iter", type=int, default=None, help=argparse.SUPPRESS)

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"Unknown arguments for official MotionGPT3 wrapper: {' '.join(unknown)}")
    if args.eval_every_step is None and args.eval_iter is not None:
        args.eval_every_step = args.eval_iter
    if args.eval_every_step is not None and args.eval_every_step < 1:
        parser.error("--eval-every-step must be >= 1")
    if args.eval_every_epoch is not None and args.eval_every_epoch < 1:
        parser.error("--eval-every-epoch must be >= 1")
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")
    return args


@contextlib.contextmanager
def pushd(path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def resolve_under_root(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def display_path_for_load(path, root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def require_motiongpt3_root(root):
    root = Path(root).resolve()
    if not (root / "train.py").exists() or not (root / "motGPT").exists():
        raise FileNotFoundError(
            f"Official MotionGPT3 repo not found at {root}. "
            f"Clone it with: git clone https://github.com/OpenMotionLab/MotionGPT3.git {root}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_official_modules(root):
    require_motiongpt3_root(root)
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from motGPT.callback import build_callbacks
    from motGPT.config import get_module_config, instantiate_from_config
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model
    from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
    from motGPT.utils.logger import create_logger

    return {
        "pl": pl,
        "OmegaConf": OmegaConf,
        "build_callbacks": build_callbacks,
        "get_module_config": get_module_config,
        "instantiate_from_config": instantiate_from_config,
        "build_data": build_data,
        "build_model": build_model,
        "load_pretrained": load_pretrained,
        "load_pretrained_vae": load_pretrained_vae,
        "create_logger": create_logger,
    }


def copy_mean_std(src_meta, dst_meta):
    dst_meta.mkdir(parents=True, exist_ok=True)
    for filename in ("mean.npy", "std.npy"):
        src = src_meta / filename
        dst = dst_meta / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing evaluator normalization file: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)


def prepare_mean_std_compat(cfg, out_dir):
    # Official HumanML3DDataModule hard-codes Comp_v6_KLD01. This repo has the
    # same evaluator normalization under Comp_v6_KLD005, so create a tiny
    # compatibility tree for official loading without modifying official code.
    src_root = REPO_ROOT / "checkpoints"
    src_meta = src_root / "t2m" / "Comp_v6_KLD005" / "meta"
    compat_root = Path(out_dir) / "_motiongpt3_compat"
    copy_mean_std(src_meta, compat_root / "t2m" / "Comp_v6_KLD01" / "meta")
    copy_mean_std(REPO_ROOT / "checkpoints" / "kit" / "Comp_v6_KLD005" / "meta",
                  compat_root / "kit" / "Comp_v6_KLD01" / "meta")
    cfg.DATASET.HUMANML3D.MEAN_STD_PATH = str(compat_root)
    cfg.DATASET.KIT.MEAN_STD_PATH = str(compat_root)


def make_absolute_if_relative(path, root):
    if not path:
        return path
    path = Path(str(path))
    return str(path if path.is_absolute() else (root / path).resolve())


def absolutize_official_paths(cfg, root):
    if "lm" in cfg and "mot_vae_gpt2" in cfg.lm:
        model_path = cfg.lm.mot_vae_gpt2.params.model_path
        cfg.lm.mot_vae_gpt2.params.model_path = make_absolute_if_relative(model_path, root)
    if "DATASET" in cfg:
        cfg.DATASET.TASK_ROOT = make_absolute_if_relative(cfg.DATASET.TASK_ROOT, root)
        cfg.DATASET.SMPL_PATH = make_absolute_if_relative(cfg.DATASET.SMPL_PATH, root)
        cfg.DATASET.TRANSFORM_PATH = make_absolute_if_relative(cfg.DATASET.TRANSFORM_PATH, root)


def preflight_official_assets(cfg, logger):
    dataset_root = Path(cfg.DATASET.HUMANML3D.ROOT)
    if not dataset_root.exists():
        raise FileNotFoundError(f"HumanML3D dataset root does not exist: {dataset_root}")
    if not Path(cfg.DATASET.WORD_VERTILIZER_PATH).exists():
        raise FileNotFoundError(f"GloVe directory does not exist: {cfg.DATASET.WORD_VERTILIZER_PATH}")
    if cfg.DATASET.TASK_PATH and not Path(cfg.DATASET.TASK_PATH).exists():
        raise FileNotFoundError(f"MotionGPT3 instruction file does not exist: {cfg.DATASET.TASK_PATH}")
    t2m_finest = Path(cfg.METRIC.TM2T.t2m_path) / "t2m" / "text_mot_match" / "model" / "finest.tar"
    if not t2m_finest.exists():
        raise FileNotFoundError(f"T2M evaluator checkpoint does not exist: {t2m_finest}")

    lm_path = Path(cfg.lm.mot_vae_gpt2.params.model_path)
    if not lm_path.exists():
        raise FileNotFoundError(
            f"Official MotionGPT3 GPT-2 assets not found at {lm_path}. "
            f"Run `bash prepare/prepare_gpt2.sh` inside the MotionGPT3 repo or pass a config with a valid lm model_path."
        )
    if cfg.TRAIN.PRETRAINED_VAE and not Path(cfg.TRAIN.PRETRAINED_VAE).exists():
        raise FileNotFoundError(
            f"Official MotionGPT3 PRETRAINED_VAE checkpoint does not exist: {cfg.TRAIN.PRETRAINED_VAE}"
        )


def load_official_cfg(args, root, modules):
    OmegaConf = modules["OmegaConf"]
    cfg_path = args.cfg or STAGE_CONFIGS[args.mgpt3_stage]
    cfg_assets_path = args.cfg_assets

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    with pushd(root):
        cfg_assets = OmegaConf.load(display_path_for_load(resolve_under_root(cfg_assets_path, root), root))
        cfg_base = OmegaConf.load(Path(cfg_assets.CONFIG_FOLDER) / "default.yaml")
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(display_path_for_load(resolve_under_root(cfg_path, root), root)))
        if not cfg_exp.FULL_CONFIG:
            cfg_exp = modules["get_module_config"](cfg_exp, cfg_assets.CONFIG_FOLDER)
        cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    cfg.DEBUG = not args.nodebug
    cfg.FOLDER = str(Path(args.out_dir).resolve())
    cfg.NAME = args.exp_name
    cfg.DATASET.HUMANML3D.ROOT = str(Path(args.dataset_root).resolve())
    cfg.DATASET.HUMANML3D.SPLIT_ROOT = str(Path(args.dataset_root).resolve())
    cfg.DATASET.WORD_VERTILIZER_PATH = str(Path(args.glove_root).resolve())
    cfg.METRIC.TM2T.t2m_path = str((REPO_ROOT / "checkpoints").resolve())
    cfg.DATASET.TASK_PATH = str(
        Path(args.task_path).resolve()
        if args.task_path
        else (root / STAGE_TASK_FILES[args.mgpt3_stage]).resolve()
    )

    prepare_mean_std_compat(cfg, cfg.FOLDER)
    absolutize_official_paths(cfg, root)

    if args.pretrained_vae:
        cfg.TRAIN.PRETRAINED_VAE = str(Path(args.pretrained_vae).resolve())
    if args.pretrained:
        cfg.TRAIN.PRETRAINED = str(Path(args.pretrained).resolve())
    if args.resume:
        cfg.TRAIN.RESUME = str(Path(args.resume).resolve())
    if args.checkpoint:
        cfg.TEST.CHECKPOINTS = str(Path(args.checkpoint).resolve())
    if args.eval_split:
        cfg.EVAL.SPLIT = args.eval_split
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.eval_batch_size is not None:
        cfg.EVAL.BATCH_SIZE = args.eval_batch_size
        cfg.TEST.BATCH_SIZE = args.eval_batch_size
    if args.num_workers is not None:
        cfg.TRAIN.NUM_WORKERS = args.num_workers
        cfg.EVAL.NUM_WORKERS = args.num_workers
        cfg.TEST.NUM_WORKERS = args.num_workers
    if args.end_epoch is not None:
        cfg.TRAIN.END_EPOCH = args.end_epoch
    if args.eval_every_epoch is not None:
        cfg.LOGGER.VAL_EVERY_STEPS = args.eval_every_epoch
    if args.lr is not None:
        cfg.TRAIN.OPTIM.params.lr = args.lr
        if "params_diff" in cfg.TRAIN.OPTIM:
            cfg.TRAIN.OPTIM.params_diff.lr = args.lr
    if args.device is not None:
        cfg.DEVICE = args.device
    if args.accumulate_grad_batches is not None:
        cfg.TRAIN.accumulate_grad_batches = args.accumulate_grad_batches
    if args.no_wandb and "WANDB" in cfg.LOGGER:
        cfg.LOGGER.WANDB.params.project = None
        cfg.LOGGER.WANDB.params.offline = True

    if not args.keep_official_metrics:
        cfg.METRIC.TYPE = []

    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        if "WANDB" in cfg.LOGGER:
            cfg.LOGGER.WANDB.params.offline = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    return cfg


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


class OurMotionTextEvalCallback(Callback):
    def __init__(self, cfg, args, logger):
        from options.get_eval_option import get_opt
        from models.evaluator_wrapper import EvaluatorModelWrapper

        from utils.eval_bitm import (
            NLGMetricverse,
            bert_score,
            calculate_activation_statistics,
            calculate_diversity,
            calculate_frechet_distance,
            calculate_multimodality,
            calculate_R_precision,
            load_metric,
            prepare_text_metric_inputs,
        )

        super().__init__()
        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.eval_wrapper = EvaluatorModelWrapper(get_opt(args.eval_opt, torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.NLGMetricverse = NLGMetricverse
        self.load_metric = load_metric
        self.bert_score = bert_score
        self.calculate_activation_statistics = calculate_activation_statistics
        self.calculate_diversity = calculate_diversity
        self.calculate_frechet_distance = calculate_frechet_distance
        self.calculate_multimodality = calculate_multimodality
        self.calculate_R_precision = calculate_R_precision
        self.prepare_text_metric_inputs = prepare_text_metric_inputs
        self.best = {
            "fid": 1000.0,
            "top1": 0.0,
            "top2": 0.0,
            "top3": 0.0,
            "matching": 1000.0,
            "div": 1000.0,
            "bleu4": 0.0,
            "rouge_l": 0.0,
            "cider": 0.0,
            "bert_f1": 0.0,
        }
        self.last_eval_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.args.eval_every_step is None:
            return
        step = trainer.global_step
        if step <= 0 or step == self.last_eval_step or step % self.args.eval_every_step != 0:
            return
        self.run_eval(trainer, pl_module, f"step {step}")

    def on_train_epoch_end(self, trainer, pl_module):
        if self.args.eval_every_step is not None and trainer.current_epoch + 1 != self.cfg.TRAIN.END_EPOCH:
            return
        epoch = trainer.current_epoch + 1
        eval_every = self.args.eval_every_epoch or self.cfg.LOGGER.VAL_EVERY_STEPS
        if epoch % eval_every != 0 and epoch != self.cfg.TRAIN.END_EPOCH:
            return
        if self.args.eval_every_step is not None and trainer.global_step == self.last_eval_step:
            return
        self.run_eval(trainer, pl_module, f"epoch {epoch}")

    def run_eval(self, trainer, pl_module, label):
        if getattr(trainer, "world_size", 1) > 1:
            trainer.strategy.barrier()
        if not trainer.is_global_zero:
            if getattr(trainer, "world_size", 1) > 1:
                trainer.strategy.barrier()
            return

        was_training = pl_module.training
        pl_module.eval()
        self.logger.info(f"Running our Motion/Text eval at {label}.")
        dataloader = trainer.datamodule.val_dataloader()
        t2m_metrics = self.eval_t2m(pl_module, dataloader, trainer)
        m2t_metrics = self.eval_m2t(pl_module, dataloader, trainer)

        metrics = {f"OurEval/{key}": value for key, value in {**t2m_metrics, **m2t_metrics}.items()}
        metrics["OurEval/global_step"] = float(trainer.global_step)
        for pl_logger in trainer.loggers:
            pl_logger.log_metrics(metrics, step=trainer.global_step)
        self.last_eval_step = trainer.global_step

        if was_training:
            pl_module.train()
        if getattr(trainer, "world_size", 1) > 1:
            trainer.strategy.barrier()
        return metrics

    def eval_t2m(self, model, dataloader, trainer):
        motion_annotation_list = []
        motion_pred_list = []
        motion_multimodality = []
        r_precision_real = 0.0
        r_precision = 0.0
        matching_score_real = 0.0
        matching_score_pred = 0.0
        nb_sample = 0

        device = model.device
        for batch in tqdm(dataloader, position=1, leave=False, desc="our t2m eval"):
            batch = move_to_device(batch, device)
            lengths = torch.as_tensor(batch["length"], device=device).long()
            bs = len(lengths)

            motion_multimodality_batch = []
            for repeat_idx in range(self.args.t2m_repeats):
                with torch.no_grad():
                    old_task_path = model.hparams.cfg.DATASET.TASK_PATH
                    try:
                        model.hparams.cfg.DATASET.TASK_PATH = ""
                        rs_set = model.val_t2m_forward(batch)
                    finally:
                        model.hparams.cfg.DATASET.TASK_PATH = old_task_path

                et_pred, em_pred = self.eval_wrapper.get_co_embeddings(
                    batch["word_embs"],
                    batch["pos_ohot"],
                    batch["text_len"],
                    rs_set["m_rst"],
                    lengths,
                )
                motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

                if repeat_idx == 0:
                    et, em = self.eval_wrapper.get_co_embeddings(
                        batch["word_embs"],
                        batch["pos_ohot"],
                        batch["text_len"],
                        rs_set["m_ref"],
                        lengths,
                    )
                    motion_annotation_list.append(em)
                    motion_pred_list.append(em_pred)

                    temp_r, temp_match = self.calculate_R_precision(
                        et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
                    )
                    r_precision_real += temp_r
                    matching_score_real += temp_match

                    temp_r, temp_match = self.calculate_R_precision(
                        et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
                    )
                    r_precision += temp_r
                    matching_score_pred += temp_match
                    nb_sample += bs

            motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        gt_mu, gt_cov = self.calculate_activation_statistics(motion_annotation_np)
        mu, cov = self.calculate_activation_statistics(motion_pred_np)

        if nb_sample > 1:
            diversity_times = 300 if nb_sample > 300 else min(100, nb_sample - 1)
            diversity_real = self.calculate_diversity(motion_annotation_np, diversity_times)
            diversity = self.calculate_diversity(motion_pred_np, diversity_times)
        else:
            diversity_real = 0.0
            diversity = 0.0
        r_precision_real = r_precision_real / nb_sample
        r_precision = r_precision / nb_sample
        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample
        fid = self.calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        multimodality = 0.0
        if self.args.t2m_repeats > 1:
            motion_multimodality_np = torch.cat(motion_multimodality, dim=0).cpu().numpy()
            mm_times = min(10, self.args.t2m_repeats - 1)
            multimodality = self.calculate_multimodality(motion_multimodality_np, mm_times)

        self.best["fid"] = min(self.best["fid"], float(fid))
        self.best["top1"] = max(self.best["top1"], float(r_precision[0]))
        self.best["top2"] = max(self.best["top2"], float(r_precision[1]))
        self.best["top3"] = max(self.best["top3"], float(r_precision[2]))
        self.best["matching"] = min(self.best["matching"], float(matching_score_pred))
        self.best["div"] = float(diversity) if abs(diversity_real - diversity) < abs(diversity_real - self.best["div"]) else self.best["div"]

        metrics = {
            "FID": float(fid),
            "Diversity": float(diversity),
            "R_precision_top_1": float(r_precision[0]),
            "R_precision_top_2": float(r_precision[1]),
            "R_precision_top_3": float(r_precision[2]),
            "Matching_score": float(matching_score_pred),
            "Multimodality": float(multimodality),
        }
        self.logger.info(
            "Our T2M eval: "
            + ", ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            + f"; real_R={r_precision_real}, real_matching={matching_score_real:.5f}"
        )
        return {f"T2M/{key}": value for key, value in metrics.items()}

    def eval_m2t(self, model, dataloader, trainer):
        if self.NLGMetricverse is None or self.load_metric is None:
            raise ImportError("Our M2T eval requires nlgmetricverse.")
        if self.bert_score is None:
            raise ImportError("Our M2T eval requires bert_score.")

        nlg_evaluator = self.NLGMetricverse([
            self.load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
            self.load_metric("bleu", resulting_name="bleu_2", compute_kwargs={"max_order": 2}),
            self.load_metric("bleu", resulting_name="bleu_3", compute_kwargs={"max_order": 3}),
            self.load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
            self.load_metric("rouge"),
            self.load_metric("cider"),
        ])

        all_pred_text = []
        all_reference_texts = []
        device = model.device
        for batch in tqdm(dataloader, position=2, leave=False, desc="our m2t eval"):
            batch = move_to_device(batch, device)
            for repeat_idx in range(self.args.m2t_repeats):
                with torch.no_grad():
                    rs_set = model.val_m2t_forward(batch)
                if repeat_idx == 0:
                    all_pred_text.extend(rs_set["t_pred"])
                    all_reference_texts.extend(rs_set["t_ref"])

        metric_pred_text, metric_reference_texts, skipped_empty = self.prepare_text_metric_inputs(
            all_pred_text, all_reference_texts
        )
        if skipped_empty:
            self.logger.info(f"Our M2T eval skipped {skipped_empty} empty prediction/reference samples.")

        if len(metric_pred_text) == 0:
            bleu1 = bleu2 = bleu3 = bleu4 = rouge_l = cider_score = bert_f1 = 0.0
        else:
            scores = nlg_evaluator(predictions=metric_pred_text, references=metric_reference_texts)
            bleu1 = scores["bleu_1"]["score"]
            bleu2 = scores["bleu_2"]["score"]
            bleu3 = scores["bleu_3"]["score"]
            bleu4 = scores["bleu_4"]["score"]
            rouge_l = scores["rouge"]["rougeL"]
            cider_score = scores["cider"]["score"]
            _, _, bert_f1_tensor = self.bert_score(
                metric_pred_text,
                metric_reference_texts,
                lang="en",
                rescale_with_baseline=True,
                idf=True,
                nthreads=0,
                verbose=False,
            )
            bert_f1 = bert_f1_tensor.mean().item()

        self.best["bleu4"] = max(self.best["bleu4"], float(bleu4))
        self.best["rouge_l"] = max(self.best["rouge_l"], float(rouge_l))
        self.best["cider"] = max(self.best["cider"], float(cider_score))
        self.best["bert_f1"] = max(self.best["bert_f1"], float(bert_f1))

        metrics = {
            "BLEU1": float(bleu1),
            "BLEU2": float(bleu2),
            "BLEU3": float(bleu3),
            "BLEU4": float(bleu4),
            "ROUGE_L": float(rouge_l),
            "CIDEr": float(cider_score),
            "BERT_F1": float(bert_f1),
        }
        self.logger.info("Our M2T eval: " + ", ".join(f"{key}={value:.5f}" for key, value in metrics.items()))
        return {f"M2T/{key}": value for key, value in metrics.items()}


def warn_legacy_args(args, logger):
    if args.vq_name:
        logger.warning("--vq-name is ignored: official MotionGPT3 uses continuous MotionVAE, not this repo's VQ-VAE.")
    if args.block_size:
        logger.warning("--block-size is ignored: sequence lengths come from official MotionGPT3 config.")
    if args.total_iter:
        logger.warning("--total-iter is ignored: official MotionGPT3 training is epoch-based; use --end-epoch.")
    if args.eval_iter:
        logger.warning("--eval-iter is treated as --eval-every-step for our evaluation callback.")


class EvalOnlyTrainerShim:
    def __init__(self, datamodule, loggers, global_step=0):
        self.datamodule = datamodule
        self.loggers = loggers
        self.global_step = global_step
        self.current_epoch = 0
        self.world_size = 1
        self.is_global_zero = True


def select_eval_device(cfg):
    if str(cfg.ACCELERATOR).lower() == "gpu" and torch.cuda.is_available():
        device_id = int(cfg.DEVICE[0]) if cfg.DEVICE else 0
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def run_eval_only(cfg, args, modules, model, datamodule, pl_loggers, logger):
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"MotionGPT3 checkpoint does not exist: {checkpoint}")

    cfg.TEST.CHECKPOINTS = str(checkpoint)
    if args.eval_split:
        cfg.EVAL.SPLIT = args.eval_split

    modules["load_pretrained"](cfg, model, logger, phase="test")
    device = select_eval_device(cfg)
    model.to(device)

    trainer = EvalOnlyTrainerShim(datamodule, pl_loggers)
    model.trainer = trainer
    callback = OurMotionTextEvalCallback(cfg, args, logger)
    logger.info(f"Running eval-only with our Motion/Text eval on split {cfg.EVAL.SPLIT}.")
    metrics = callback.run_eval(trainer, model, f"checkpoint {checkpoint.name}")

    output_path = Path(cfg.FOLDER_EXP) / f"our_eval_{cfg.EVAL.SPLIT}_{cfg.TIME}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info(f"Our eval-only metrics are saved to {output_path}")
    return metrics


def main():
    args = parse_args()
    root = require_motiongpt3_root(args.motiongpt3_root)
    modules = load_official_modules(root)
    pl = modules["pl"]
    OmegaConf = modules["OmegaConf"]

    cfg = load_official_cfg(args, root, modules)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger = modules["create_logger"](cfg, phase="train")
    logger.info(OmegaConf.to_yaml(cfg))
    warn_legacy_args(args, logger)
    preflight_official_assets(cfg, logger)

    pl.seed_everything(cfg.SEED_VALUE)

    pl_loggers = []
    for logger_name in cfg.LOGGER.TYPE:
        if logger_name == "tensorboard" or cfg.LOGGER.WANDB.params.project:
            pl_logger = modules["instantiate_from_config"](eval(f"cfg.LOGGER.{logger_name.upper()}"))
            pl_loggers.append(pl_logger)

    callbacks = []
    if not args.eval_only:
        callbacks = modules["build_callbacks"](cfg, logger=logger, phase="train")
        if not args.skip_our_eval:
            callbacks.append(OurMotionTextEvalCallback(cfg, args, logger))
        logger.info("Callbacks initialized")
    else:
        logger.info("Eval-only mode: skipping official training callbacks.")

    datamodule = modules["build_data"](cfg)
    logger.info("datasets module {} initialized".format("".join(cfg.DATASET.target.split(".")[-2])))

    model = modules["build_model"](cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    trainer = None
    if not args.eval_only:
        trainer_kwargs = {
            "default_root_dir": cfg.FOLDER_EXP,
            "max_epochs": cfg.TRAIN.END_EPOCH,
            "logger": pl_loggers,
            "callbacks": callbacks,
            "check_val_every_n_epoch": cfg.LOGGER.VAL_EVERY_STEPS,
            "accelerator": cfg.ACCELERATOR,
            "devices": cfg.DEVICE,
            "num_nodes": cfg.NUM_NODES,
            "strategy": "ddp_find_unused_parameters_true" if len(cfg.DEVICE) > 1 else "auto",
            "benchmark": False,
            "deterministic": False,
            "accumulate_grad_batches": cfg.TRAIN.accumulate_grad_batches,
        }
        if not args.keep_official_val:
            trainer_kwargs["limit_val_batches"] = 0
        trainer = pl.Trainer(**trainer_kwargs)
        logger.info("Trainer initialized")

    if cfg.TRAIN.PRETRAINED and not args.eval_only:
        modules["load_pretrained"](cfg, model, logger)
    if cfg.TRAIN.PRETRAINED_VAE:
        modules["load_pretrained_vae"](cfg, model, logger)
    else:
        logger.warning("No PRETRAINED_VAE set. Official MotionGPT3 LM training normally requires it.")

    if args.eval_only:
        run_eval_only(cfg, args, modules, model, datamodule, pl_loggers, logger)
        logger.info(f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
        logger.info("Eval-only ends!")
        return

    if cfg.TRAIN.RESUME:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    logger.info(f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
