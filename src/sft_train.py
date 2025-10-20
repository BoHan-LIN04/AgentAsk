import os, yaml, argparse, random, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import ClarifyDataset, collate_fn

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    return ap.parse_args()

def compute_sft_loss(logits, labels, weights, meta_batch, tokenizer, loss_cfg):

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = weights[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    ce_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())

    device = logits.device
    type_loss = torch.tensor(0.0, device=device, requires_grad=True)
    ask_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_samples = len(meta_batch)

    if total_samples == 0:
        basic_loss = (ce_loss * shift_weights).mean()
        return basic_loss, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    for batch_idx, meta in enumerate(meta_batch):
        target_json = meta.get('target_json', {})
        need_clarify = target_json.get('need_clarify', False)

        sample_ce = ce_loss[batch_idx]
        sample_weights = shift_weights[batch_idx]
        sample_labels = shift_labels[batch_idx]

        need_clarify_tokens = find_token_span(
            tokenizer, sample_labels, 
            f'"need_clarify": {"true" if need_clarify else "false"}'
        )
        
        if need_clarify_tokens:

            type_weight_factor = loss_cfg.get('type_weight_factor', 1.0)
            for pos in need_clarify_tokens:
                if pos < len(sample_ce) and sample_labels[pos] != -100:
                    type_loss = type_loss + sample_ce[pos] * type_weight_factor

        if need_clarify:
            agent_weight_factor = loss_cfg.get('agent_weight_factor', 1.0)

            to_agent = target_json.get('to_agent')
            if to_agent:
                agent_tokens = find_token_span(
                    tokenizer, sample_labels, f'"to_agent": "{to_agent}"'
                )
                for pos in agent_tokens:
                    if pos < len(sample_ce) and sample_labels[pos] != -100:
                        ask_loss = ask_loss + sample_ce[pos] * agent_weight_factor

            clarify_q = target_json.get('clarify_question')
            if clarify_q:
                question_tokens = find_token_span(
                    tokenizer, sample_labels, f'"clarify_question": "{clarify_q}"'
                )
                for pos in question_tokens:
                    if pos < len(sample_ce) and sample_labels[pos] != -100:
                        ask_loss = ask_loss + sample_ce[pos]

            ask_type = target_json.get('type')
            if ask_type:
                type_tokens = find_token_span(
                    tokenizer, sample_labels, f'"type": "{ask_type}"'
                )
                for pos in type_tokens:
                    if pos < len(sample_ce) and sample_labels[pos] != -100:
                        ask_loss = ask_loss + sample_ce[pos]

    type_loss = type_loss / total_samples if total_samples > 0 else type_loss
    ask_loss = ask_loss / total_samples if total_samples > 0 else ask_loss

    lambda_ask = loss_cfg.get('lambda_ask', 1.0)
    total_loss = type_loss + lambda_ask * ask_loss

    if total_loss.item() == 0.0:
        total_loss = (ce_loss * shift_weights).mean()
    
    return total_loss, type_loss, ask_loss

def find_token_span(tokenizer, token_ids, text_fragment):
    fragment_tokens = tokenizer.encode(text_fragment, add_special_tokens=False)
    if not fragment_tokens:
        return []
    positions = []
    token_list = token_ids.tolist() if hasattr(token_ids, 'tolist') else token_ids
    
    for i in range(len(token_list) - len(fragment_tokens) + 1):
        if token_list[i:i+len(fragment_tokens)] == fragment_tokens:
            positions.extend(range(i, i + len(fragment_tokens)))
    
    return positions

def main():
    args=parse()
    cfg=yaml.safe_load(open(args.config))
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    tok=AutoTokenizer.from_pretrained(cfg['model_name'])
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(
        cfg['model_name'],
        load_in_8bit=cfg.get('load_in_8bit',False),
        device_map="auto"
    )
    lcfg=LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        target_modules=cfg['lora']['target_modules'],
        task_type="CAUSAL_LM"
    )
    model=get_peft_model(model,lcfg)

    train_path = cfg.get('train_data',"data/clarify_samples.jsonl")
    loss_cfg = cfg.get('loss', {})
    type_w = float(loss_cfg.get('type_weight_factor', 3.0))
    agent_w = float(loss_cfg.get('agent_weight_factor', 3.0))
    need_w = float(loss_cfg.get('need_clarify_weight_factor', 1.0))
    ds_cfg = cfg.get('dataset', {})
    balance_need = bool(ds_cfg.get('balance_need', True))

    ds=ClarifyDataset(
        train_path, tok, max_len=cfg['max_len'],
        type_weight=type_w, agent_weight=agent_w, need_weight=need_w,
        balance_need=balance_need
    )
    dl=DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b,tok.pad_token_id)
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable_params,lr=float(cfg['lr']))
    steps=len(dl)*cfg['epochs']
    sched=get_linear_schedule_with_warmup(opt,cfg['warmup_steps'],steps)

    model.train()
    for ep in range(cfg['epochs']):
        bar=tqdm(dl,desc=f"SFT-epoch{ep}")
        epoch_type_loss = 0.0
        epoch_ask_loss = 0.0
        
        for batch in bar:
            for k in ["input_ids","attention_mask","labels","weights"]:
                batch[k]=batch[k].to(model.device)
            out=model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss_cfg = cfg.get('loss', {})
            total_loss, type_loss, ask_loss = compute_sft_loss(
                out.logits, 
                batch["labels"], 
                batch["weights"], 
                batch["meta"], 
                tok, 
                loss_cfg
            )

            total_loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            epoch_type_loss += float(type_loss.item())
            epoch_ask_loss += float(ask_loss.item())
            
            bar.set_postfix(
                total=f"{float(total_loss.item()):.4f}",
                type=f"{float(type_loss.item()):.4f}",
                ask=f"{float(ask_loss.item()):.4f}"
            )

    os.makedirs(cfg['output_dir'],exist_ok=True)
    model.save_pretrained(cfg['output_dir'])
    tok.save_pretrained(cfg['output_dir'])
    print("SFT done.")


if __name__=="__main__":
    main()
