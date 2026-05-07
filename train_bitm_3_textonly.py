import codecs as cs
import json
import os
import warnings
from os.path import join as pjoin

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

import options.option_transformer as option_trans
import utils.utils_model as utils_model
from dataset import dataset_TM_train
from exit.utils import get_model, init_save_folder, maybe_data_parallel

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CaptionOnlyDataset(Dataset):
    def __init__(self, dataset_name, split='train'):
        if dataset_name == 't2m':
            data_root = './dataset/HumanML3D'
        elif dataset_name == 'kit':
            data_root = './dataset/KIT-ML'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        split_file = pjoin(data_root, f'{split}.txt')
        text_dir = pjoin(data_root, 'texts')
        captions = []

        with cs.open(split_file, 'r') as f:
            id_list = [line.strip() for line in f.readlines() if line.strip()]

        for name in tqdm(id_list, desc=f'Loading {split} captions'):
            text_path = pjoin(text_dir, name + '.txt')
            try:
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        fields = line.strip().split('#')
                        if len(fields) < 1:
                            continue
                        caption = fields[0].strip()
                        if caption:
                            captions.append(caption)
            except OSError:
                continue

        if len(captions) == 0:
            raise RuntimeError(f"No captions found for dataset={dataset_name}, split={split}.")
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, item):
        return self.captions[item]


##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

init_save_folder(args)
os.makedirs(args.out_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- BERT MLM ---- #####
bert_name = 'google-bert/bert-large-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_name)
text_model = AutoModelForMaskedLM.from_pretrained(bert_name)

special_ids_t = {
    'mask_id': tokenizer.mask_token_id,
    'cls_id': tokenizer.cls_token_id,
    'eos_id': tokenizer.sep_token_id,
    'pad_id': tokenizer.pad_token_id
}

if args.resume_trans is not None:
    print('loading text checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    state_dict = ckpt.get('bert_mlm', ckpt.get('model', ckpt))
    text_model.load_state_dict(state_dict, strict=True)

text_model.to(device)
text_model.train()
text_model, parallel_info = maybe_data_parallel(
    text_model,
    batch_size=args.batch_size,
    min_batch_per_gpu=args.min_batch_per_gpu,
    logger=logger
)
trainable_params = sum(param.numel() for param in get_model(text_model).parameters() if param.requires_grad)
logger.info(f'Text-only BERT MLM training: {trainable_params:,} params trainable.')
logger.info(
    f"Runtime parallelism: visible_gpus={parallel_info['visible_gpus']}, "
    f"used_gpus={parallel_info['used_gpus']}, "
    f"batch_per_gpu={parallel_info['batch_per_gpu']:.1f}, "
    f"data_parallel={parallel_info['data_parallel']}."
)

##### ---- Optimizer & Scheduler ---- #####
if args.decay_option == 'noVQ':
    logger.warning("Text-only training has no VQ layer; using decay_option='all' instead of 'noVQ'.")
    args.decay_option = 'all'
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, text_model, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Dataloader ---- #####
train_loader = DataLoader(
    CaptionOnlyDataset(args.dataname, split='train'),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True
)
train_loader_iter = dataset_TM_train.cycle(train_loader)
val_loader = DataLoader(
    CaptionOnlyDataset(args.dataname, split='val'),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    drop_last=False
)


def tokenize_text(text):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=args.max_t,
        return_tensors='pt'
    )


def masking_text(input_ids, attention_mask, probs=(0, 1)):
    curr_device = input_ids.device
    special_mask = (
        (input_ids == special_ids_t['cls_id'])
        | (input_ids == special_ids_t['eos_id'])
        | (input_ids == special_ids_t['pad_id'])
    )
    valid_mask = attention_mask.bool() & ~special_mask

    if probs[0] == 0 and probs[1] == 0:
        mask_token = torch.zeros_like(input_ids, dtype=torch.bool)
    else:
        rand_probs = (probs[1] - probs[0]) * torch.rand(input_ids.shape[0], 1, device=curr_device) + probs[0]
        mask_token = (torch.rand(input_ids.shape, device=curr_device) < rand_probs) & valid_mask

    masked_input_ids = input_ids.masked_fill(mask_token, special_ids_t['mask_id'])
    return masked_input_ids, valid_mask, mask_token


def get_loss(logits, target, loss_mask):
    batch_size, _, vocab_size = logits.shape
    target = target.long()
    flat_mask = loss_mask.reshape(-1)
    if not flat_mask.any():
        return logits.new_tensor(0.0)

    pred_masked = logits.reshape(-1, vocab_size)[flat_mask]
    target_masked = target.reshape(-1)[flat_mask]
    ce_masked = F.cross_entropy(pred_masked, target_masked, reduction='none')

    denom = loss_mask.sum(dim=1, keepdim=True).clamp(min=1) * batch_size
    weights = (loss_mask.float() / denom).reshape(-1)[flat_mask]
    return (ce_masked * weights).sum()


def get_acc(logits, target, mask):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    active_logits = logits[mask]
    active_targets = target[mask]
    predictions = active_logits.argmax(dim=-1)
    correct = (predictions == active_targets).float().sum()
    return (correct / mask.sum()) * 100


@torch.no_grad()
def evaluate(nb_iter, mask_probs=(0, 1)):
    text_model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0

    for text in tqdm(val_loader, position=1, leave=False):
        text_inputs = tokenize_text(text)
        token_ids_t = text_inputs['input_ids'].to(device)
        seq_mask_t = text_inputs['attention_mask'].to(device)
        masked_input_ids_t, _, mask_token_t = masking_text(token_ids_t, seq_mask_t, probs=mask_probs)
        if not mask_token_t.any():
            continue

        outputs = text_model(input_ids=masked_input_ids_t, attention_mask=seq_mask_t, return_dict=True)
        loss = get_loss(outputs.logits, token_ids_t, mask_token_t)
        acc = get_acc(outputs.logits, token_ids_t, mask_token_t)

        total_loss += loss.item()
        total_acc += acc.item()
        total_steps += 1

    mean_loss = total_loss / total_steps if total_steps > 0 else 0.0
    mean_acc = total_acc / total_steps if total_steps > 0 else 0.0

    writer.add_scalar('./Val/text_masked_loss', mean_loss, nb_iter)
    writer.add_scalar('./Val/text_masked_acc', mean_acc, nb_iter)
    logger.info(f"Val. Iter {nb_iter}: text_masked_loss {mean_loss:.6f}, text_masked_acc {mean_acc:.4f}")

    text_model.train()
    return mean_loss, mean_acc


def save_checkpoint(filename):
    torch.save(
        {
            'bert_mlm': get_model(text_model).state_dict(),
            'bert_name': bert_name,
            'args': vars(args),
        },
        os.path.join(args.out_dir, filename)
    )


def train(mask_probs=(0, 1)):
    best_val_loss = float('inf')

    for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
        text = next(train_loader_iter)
        text_inputs = tokenize_text(text)
        token_ids_t = text_inputs['input_ids'].to(device)
        seq_mask_t = text_inputs['attention_mask'].to(device)
        masked_input_ids_t, _, mask_token_t = masking_text(token_ids_t, seq_mask_t, probs=mask_probs)
        if not mask_token_t.any():
            continue

        outputs = text_model(input_ids=masked_input_ids_t, attention_mask=seq_mask_t, return_dict=True)
        loss = get_loss(outputs.logits, token_ids_t, mask_token_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if nb_iter % args.print_iter == 0:
            masked_acc = get_acc(outputs.logits, token_ids_t, mask_token_t)
            writer.add_scalar('./Loss/text_masked', loss, nb_iter)
            writer.add_scalar('./ACC/text_masked', masked_acc, nb_iter)
            writer.add_scalar('./Mask/text_masked_ratio', mask_token_t.float().mean(), nb_iter)
            logger.info(
                f"Train. Iter {nb_iter}: text_masked_loss {loss.item():.6f}, "
                f"text_masked_acc {masked_acc.item():.4f}, mask_ratio {mask_token_t.float().mean().item():.4f}"
            )

        if nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
            val_loss, _ = evaluate(nb_iter, mask_probs=mask_probs)
            save_checkpoint('net_last.pth')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint('net_best_text.pth')
                logger.info(f"Best text checkpoint updated at iter {nb_iter} with val loss {best_val_loss:.6f}.")


train(mask_probs=(0, 1))
