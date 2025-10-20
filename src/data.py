import json, random
from torch.utils.data import Dataset
from .templates import build_input, target_json_str

class ClarifyDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024, type_weight=3.0, agent_weight=2.5, need_weight=1.5, balance_need=True):
        self.items=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                ex=json.loads(line)
                if 'target_json' not in ex:
                    tj = self._build_min_target_json(ex)
                    if tj is None:
                        continue
                    ex['target_json'] = tj
                self.items.append(ex)
        if balance_need and self.items:
            pos=[e for e in self.items if bool(e['target_json'].get('need_clarify'))]
            neg=[e for e in self.items if not bool(e['target_json'].get('need_clarify'))]
            if pos and neg:
                if len(pos)>len(neg):
                    neg = neg * (len(pos)//len(neg)) + random.sample(neg, len(pos)%len(neg))
                elif len(neg)>len(pos):
                    pos = pos * (len(neg)//len(pos)) + random.sample(pos, len(neg)%len(pos))
                self.items = pos+neg
        random.shuffle(self.items)
        self.tok=tokenizer
        self.max_len=max_len
        self.type_weight = float(type_weight)
        self.agent_weight = float(agent_weight)
        self.need_weight = float(need_weight)

    def _build_min_target_json(self, ex):
        need = bool(ex.get('clarify_question')) or bool(ex.get('issues'))
        to_agent = ex.get('to_agent')
        typ = ex.get('type')
        cq = ex.get('clarify_question') or ""
        if not need and not (to_agent or typ or cq):
            return None
        return {
            "need_clarify": need,
            "to_agent": to_agent,
            "type": typ,
            "clarify_question": cq
        }

    def __len__(self): return len(self.items)

    def __getitem__(self,idx):
        ex=self.items[idx]
        prompt=build_input(ex)
        tgt=target_json_str(ex['target_json'])
        full=prompt+" "+tgt


        enc_full = self.tok(full, add_special_tokens=False, truncation=True, max_length=self.max_len)
        enc_prompt = self.tok(prompt, add_special_tokens=False, truncation=True, max_length=self.max_len)

        ids = enc_full['input_ids']
        prompt_len = min(len(enc_prompt['input_ids']), len(ids))

        attn = [1]*len(ids)

        labels = [-100]*prompt_len + ids[prompt_len:]

        weights = [0.0]*prompt_len + [1.0]*(len(ids)-prompt_len)

        frag_fac = []


        need_val = ex['target_json'].get('need_clarify')
        if need_val is not None:
            need_str = f'"need_clarify": {"true" if bool(need_val) else "false"}'
            frag_fac.append((need_str, self.need_weight))


        typ = ex['target_json'].get('type')
        if typ:
            frag_fac.append((f'"type": "{typ}"', self.type_weight))


        agent = ex['target_json'].get('to_agent')
        if agent:
            frag_fac.append((f'"to_agent": "{agent}"', self.agent_weight))

        for frag, fac in frag_fac:
            kid = self.tok(frag, add_special_tokens=False)['input_ids']
            if not kid: continue
            start = prompt_len
            end = len(ids)
            for i in range(start, max(start, end - len(kid)) + 1):
                if ids[i:i+len(kid)] == kid:
                    for j in range(i, min(i+len(kid), end)):
                        weights[j] *= fac

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "weights": weights,
            "target_json": ex['target_json'],
        }

def collate_fn(batch,pad_id):
    import torch
    m=max(len(b['input_ids']) for b in batch)
    def pad_list(x, fill): return x + [fill]*(m - len(x))
    ids=torch.tensor([pad_list(b['input_ids'], pad_id) for b in batch], dtype=torch.long)
    att=torch.tensor([pad_list(b['attention_mask'], 0) for b in batch], dtype=torch.long)
    lab=torch.tensor([pad_list(b['labels'], -100) for b in batch], dtype=torch.long)
    w=torch.tensor([pad_list(b['weights'], 0.0) for b in batch], dtype=torch.float)
    return dict(input_ids=ids,attention_mask=att,labels=lab,weights=w,meta=batch)
