import json
import re
import uuid
from pathlib import Path
from datetime import datetime
from loguru import logger
import os, re, json, uuid
import torch
import torch.nn.functional as F
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI as OpenAIClient  
    openai_legacy = None
except Exception:
    OpenAIClient = None
    try:
           
           import openai as openai_legacy
    except Exception:
        openai_legacy = None

class ClarifyManager:

    def __init__(self, enabled: bool=False, mode: str="llm", model: str="gpt-4o-mini", student_model_path: str=None,
                 output_path: str="clarify_requests.jsonl", timeout: int=15, task: str = "gsm8k", prompt_dir: str = "MAR/ClarifyPrompts"):
        self.enabled = enabled
        self.mode = mode or "llm"
        self.model = model
        self.student_model_path = student_model_path
        self._student_tokenizer = None
        self._student_model = None
        self._student_loaded = False
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.touch(exist_ok=True)
        self.timeout = timeout
        self.task = task
        self.prompt_dir = Path(prompt_dir)
        self._last_fmt_ok: bool = False
        self._last_q_len: int = 0
        self.online_rl: bool = False
        self.rl_lr: float = 1e-5
        self._rl_opt = None
        self._last_inputs_ids = None
        self._last_out_ids = None
        self.rl_train_head_only: bool = True
        self.rl_train_last_n_layers: int = 0

        self.alpha_eff: float = 1.0     
        self.alpha_fmt: float = 0.5     
        self.alpha_ans: float = 2.0     
        self.lambda_sw: float = 0.2     
        self.lambda_R: float = 1.0      
        self.beta: float = 0.01         
        self.epsilon: float = 0.2      
        self.H: int = 5                  
        self.num_samples: int = 4        
        self.sample_temperature: float = 1.0  
        

        self._sliding_window = deque(maxlen=self.H)
        self._episode_edges: List[Dict] = []  
        self._reference_model = None
        self._old_policy_cache: Dict = {}
        self._current_trajectories: List[Dict] = []  

    def new_request_id(self):
        return f"clarify_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def _read_env(self):
        env_key = None
        env_url = None
        try:
            env_path = Path(__file__).resolve().parents[1] / '.env'
            if env_path.exists():
                for L in env_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                    if '=' not in L:
                        continue
                    k, v = L.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('\'"')
                    if k.upper().strip() in ('KEY', 'API_KEY', 'OPENAI_KEY'):
                        env_key = v
                    if k.upper().strip() in ('URL', 'API_BASE', 'OPENAI_API_BASE'):
                        env_url = v
        except Exception as e:
            logger.debug(f"ClarifyManager: Failed to read .env: {e}")
        return env_key, env_url

    def _load_template(self):
        try:
            p = self.prompt_dir / f"{self.task}.json"
            if p.exists():
                return json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            logger.debug(f"ClarifyManager: Failed to load prompt template for task {self.task}: {e}")
        return {
            "system": "You are a concise reviewer. Return ONLY valid JSON with at least needs_clarify(bool) and clarify_question(str).",
            "user": ("Context:\n{system_roles_desc}\nTopology:\n{topology_desc}\n"
                     "Current role: {current_agent_role}\nNext roles: {next_agent_roles}\n"
                     "Role prompt: {role_prompt}\nAgent output:\n{output_text}\n\n"
                     "Check: intermediate steps, arithmetic/logic, final answer format. If clarification needed produce a single closed-ended question (≤25 words).")
        }

    def _call_llm_raw(self, system: str, user: str):
        env_key, env_url = self._read_env()
        import re, json
        if OpenAIClient is not None:
            try:
                try:
                    client = OpenAIClient(api_key=env_key, api_base=env_url.rstrip('/') if env_url else None)
                except Exception:
                    client = OpenAIClient()
                logger.debug(f"ClarifyManager: calling OpenAI.chat.completions.create model={self.model}")
                resp = client.chat.completions.create(model=self.model, messages=[{"role":"system","content":system},{"role":"user","content":user}], timeout=self.timeout)
                text = None
                try:
                    text = resp.choices[0].message.content.strip()
                except Exception:
                    jr = resp if isinstance(resp, dict) else getattr(resp, '__dict__', {})
                    text = (jr.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(jr, dict) else "")
                m = re.search(r'(\{.*\})', text, flags=re.S) if text else None
                json_text = m.group(1) if m else (text or "")
                parsed = json.loads(json_text) if json_text else None
                return parsed
            except Exception as e:
                logger.debug(f"ClarifyManager: OpenAI client call failed: {e}", exc_info=True)

        if openai_legacy is not None:
            try:
                if env_key:
                    openai_legacy.api_key = env_key
                if env_url:
                    openai_legacy.api_base = env_url.rstrip('/')
                logger.debug(f"ClarifyManager: calling legacy openai.ChatCompletion model={self.model}")
                resp = openai_legacy.ChatCompletion.create(model=self.model, messages=[{"role":"system","content":system},{"role":"user","content":user}], request_timeout=self.timeout)
                text = resp["choices"][0]["message"]["content"].strip()
                m = re.search(r'(\{.*\})', text, flags=re.S)
                json_text = m.group(1) if m else text
                parsed = json.loads(json_text)
                return parsed
            except Exception as e:
                logger.debug(f"ClarifyManager: legacy openai.ChatCompletion failed: {e}", exc_info=True)

        try:
            import requests
            if env_url is None or env_key is None:
                logger.debug("ClarifyManager: No env URL/KEY for HTTP fallback.")
                return None
            endpoint = env_url.rstrip('/') + '/chat/completions'
            headers = {"Authorization": f"Bearer {env_key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": [{"role":"system","content":system},{"role":"user","content":user}]}
            logger.debug(f"ClarifyManager: POST {endpoint} model={self.model}")
            r = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            jr = r.json()
            text = None
            if "choices" in jr and len(jr["choices"])>0:
                ch = jr["choices"][0]
                if isinstance(ch.get("message"), dict):
                    text = ch["message"].get("content", "")
                else:
                    text = ch.get("text", "")
            if not text:
                logger.debug(f"ClarifyManager: HTTP fallback response missing content: {jr}")
                return None
            m = re.search(r'(\{.*\})', text, flags=re.S)
            json_text = m.group(1) if m else text
            parsed = json.loads(json_text)
            return parsed
        except Exception as e:
            logger.debug(f"ClarifyManager: HTTP fallback failed: {e}", exc_info=True)
            return None

    def _normalize_parsed(self, parsed: dict):
        if not isinstance(parsed, dict):
            return None

        def _infer_type(p: dict):
            t = p.get("type")
            if isinstance(t, str):
                t_clean = t.strip().upper()
                if t_clean in ("DG","RD","SC","CG"):
                    return t_clean
            issues = p.get("issues", [])
            if isinstance(issues, (list, tuple)):
                for it in issues:
                    if not isinstance(it, str):
                        continue
                    up = it.upper()
                    for cand in ("DG","RD","SC","CG"):
                        if up.startswith(cand + ":") or up == cand:
                            return cand
            return ""

        if "needs_clarify" in parsed and isinstance(parsed["needs_clarify"], bool):
            out = {
                "needs_clarify": parsed["needs_clarify"],
                "clarify_question": parsed.get("clarify_question", ""),
                "issues": parsed.get("issues", []),
                "type": _infer_type(parsed)
            }
            return out

        for key in ("clarify_question", "clarifying_question", "clarification", "clarification_needed", "clarifying", "question"):
            if key in parsed:
                v = parsed[key]
                if isinstance(v, str):
                    s = v.strip()
                    lowered = s.lower()
                    if lowered in ("no clarifications needed.", "no clarifications needed", "no clarification needed", "no", "none"):
                        return {"needs_clarify": False, "clarify_question": "", "issues": parsed.get("issues", []), "type": _infer_type(parsed)}
                    if "?" in s or len(s.split()) <= 25:
                        return {"needs_clarify": True, "clarify_question": s, "issues": parsed.get("issues", []), "type": _infer_type(parsed)}
                    if len(s) > 0:
                        q_line = s.splitlines()[0][:200]
                        return {"needs_clarify": True, "clarify_question": q_line, "issues": parsed.get("issues", []), "type": _infer_type(parsed)}

        if "issues" in parsed and isinstance(parsed["issues"], (list, tuple)) and len(parsed["issues"]):
            first = parsed["issues"][0]
            q = first if isinstance(first, str) else str(first)
            return {
                "needs_clarify": True,
                "clarify_question": q.strip()[:200],
                "issues": parsed.get("issues"),
                "type": _infer_type(parsed)
            }


        tc = parsed.get("test_coverage") or parsed.get("implementation_check")
        if isinstance(tc, dict):
            failed = tc.get("failed_tests") or tc.get("tests_failed") or []
            suggested = tc.get("additional_tests_suggested") or tc.get("additional_tests") or []
            if (isinstance(failed, (list, tuple)) and len(failed)) or (isinstance(suggested, (list, tuple)) and len(suggested)):
                q = parsed.get("clarification_needed") or "Please specify additional test/boundary cases needed."
                return {
                    "needs_clarify": True,
                    "clarify_question": q,
                    "issues": parsed.get("issues", []),
                    "type": _infer_type(parsed)
                }

        return None

    def heuristic_check(self, output_text: str):
        txt = str(output_text).strip().lower()
        if len(txt) < 20:
            return {"needs_clarify": True, "clarify_question": "Please provide detailed reasoning steps and key intermediate results.", "issues": [], "type": "DG"}
        for k in ["i don't know", "don't know", "unclear", "not sure"]:
            if k in txt:
                return {"needs_clarify": True, "clarify_question": "Please explain why you cannot answer or provide missing information.", "issues": [], "type": "DG"}
        return {"needs_clarify": False, "clarify_question": "", "issues": [], "type": ""}

    def _init_student_model(self):
        if self._student_loaded:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = self.student_model_path
        if (path is None) or (os.path.isfile(path)) or (not os.path.exists(path)):
            fallback = "/home/yangkuo/clarify/outputs/sft"
            if os.path.exists(fallback):
                path = fallback
            else:
                logger.warning(f"ClarifyManager(student):no path exists {self.student_model_path}")
                self._student_loaded = True
                return
        try:
            logger.info(f"ClarifyManager(student): loading local clarify model from {path}")
            tok = AutoTokenizer.from_pretrained(path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(path)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            self._student_tokenizer = tok
            self._student_model = model
            logger.info(f"ClarifyManager(student): model loaded.")
            if self.mode == 'student' and self.online_rl:
                trainable = self._select_trainable_params()
                self._rl_opt = torch.optim.AdamW(trainable, lr=self.rl_lr)
                self._student_model.train()
            else:
                self._student_model.eval()
        except Exception as e:
            logger.error(f"ClarifyManager(student): load failed: {e}")
        self._student_loaded = True

    def _select_trainable_params(self):
        model = self._student_model

        for p in model.parameters():
            p.requires_grad = False

        trainable = []

        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True
            trainable.extend(list(head.parameters()))
            logger.info("ClarifyManager(student): train only lm_head parameters.")

        n = int(getattr(self, "rl_train_last_n_layers", 0) or 0)
        if n > 0:
            layer_lists = []
            for path in ["model.layers", "transformer.h", "layers", "encoder.layers"]:
                try:
                    obj = model
                    for attr in path.split("."):
                        obj = getattr(obj, attr)
                    if hasattr(obj, "__len__"):
                        layer_lists = list(obj)
                        break
                except Exception:
                    continue
            if layer_lists:
                sel = layer_lists[-n:]
                for layer in sel:
                    for p in layer.parameters():
                        p.requires_grad = True
                for layer in sel:
                    trainable.extend(list(layer.parameters()))
                logger.info(f"ClarifyManager(student): additionally unfreeze last {n} transformer layer(s).")
            else:
                logger.warning("ClarifyManager(student): cannot locate transformer layers; skip last-N unfreeze.")

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"ClarifyManager(student): trainable_params={num_trainable}")
        if len(trainable) == 0:
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True
                trainable = list(head.parameters())
            else:
                smallest = []
                for name, module in model.named_modules():
                    if "emb" in name.lower():
                        smallest = list(module.parameters())
                        break
                if smallest:
                    for p in smallest:
                        p.requires_grad = True
                    trainable = smallest
                    logger.warning("ClarifyManager(student): fallback to train embedding params (no lm_head found).")
        return trainable

    def _last_logprob(self):
        if self._student_model is None or self._last_inputs_ids is None or self._last_out_ids is None:
            return None
        model = self._student_model
        device = next(model.parameters()).device
        inp = self._last_inputs_ids.to(device)     
        out = self._last_out_ids.to(device)             
        full = torch.cat([inp, out], dim=1)             
        attn = torch.ones_like(full, dtype=torch.long)
        model.train()
        if torch.is_inference_mode_enabled():
            logger.warning("ClarifyManager(student): torch.inference_mode is enabled — gradients may be disabled.")
        with torch.enable_grad():
            out_all = model(input_ids=full, attention_mask=attn)
            logits = out_all.logits 
            start = full.shape[1] - out.shape[1]
            pred_logits = logits[:, start-1:full.shape[1]-1, :]   
            target = out                                          
            logprobs = torch.log_softmax(pred_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)  
            return logprobs.sum()  

    def rl_update(self, reward: float):

        if self.mode != 'student':
            return

        self.e_grpo_update(terminal_reward=reward)

    def rl_config(self, alpha_eff: float = None, alpha_fmt: float = None, alpha_ans: float = None,
                  lambda_sw: float = None, lambda_R: float = None, beta: float = None,
                  epsilon: float = None, H: int = None, num_samples: int = None, 
                  sample_temperature: float = None):
        if alpha_eff is not None:
            self.alpha_eff = float(alpha_eff)
        if alpha_fmt is not None:
            self.alpha_fmt = float(alpha_fmt)
        if alpha_ans is not None:
            self.alpha_ans = float(alpha_ans)
        if lambda_sw is not None:
            self.lambda_sw = float(lambda_sw)
        if lambda_R is not None:
            self.lambda_R = float(lambda_R)
        if beta is not None:
            self.beta = float(beta)
        if epsilon is not None:
            self.epsilon = float(epsilon)
        if H is not None and H > 0:
            self.H = int(H)
            self._sliding_window = deque(list(self._sliding_window), maxlen=self.H)
        if num_samples is not None and num_samples > 0:
            self.num_samples = int(num_samples)
        if sample_temperature is not None:
            self.sample_temperature = float(sample_temperature)

    def reset_episode(self):
        if self.mode != 'student':
            return
        self._sliding_window.clear()
        self._episode_edges.clear()
        self._old_policy_cache.clear()

    def compute_edge_reward(self, z_t: bool, n_next: bool, fmt_ok: bool) -> float:
        if self.mode != 'student':
            return 0.0

        if z_t and not n_next:
            r_eff = 1.0
        elif z_t and n_next:
            r_eff = -1.0
        else:
            r_eff = 0.0

        self._sliding_window.append(1 if z_t else 0)
        c_t = sum(self._sliding_window)
        r_par = -self.lambda_sw * max(c_t - 1, 0)
        r_fmt = self.alpha_fmt * (1.0 if (z_t and fmt_ok) else 0.0)
        r_edge = self.alpha_eff * r_eff + r_par + r_fmt
        
        return r_edge

    def _student_clarify_multi_sample(self, output_text: str) -> List[Dict]:

        self._init_student_model()
        if self._student_model is None:
            return []
            
        tok = self._student_tokenizer
        model = self._student_model
        instr = (
            "You are a result compliance and information sufficiency evaluator. Given an agent's output, determine if clarification is needed from it or upstream.\n"
            "Return only JSON with fields: need_clarify(bool), to_agent(str|null), type(str|DG|RD|SC|CG|null), clarify_question(str|null).\n"
            "Rules:\n"
            "1) If output is placeholder, error, very short(<20 chars) or lacks necessary values/conclusions, set need_clarify=True and provide a question <=25 words.\n"
            "2) Map type based on question: Data Gap=DG, Reasoning Defect=RD, Insufficient Steps=SC, Format/Answer Non-compliance=CG.\n"
            "3) If no clarification needed, need_clarify=false and other fields null.\n"
            "4) Strict JSON only, no extra text."
        )
        prompt = f"{instr}\n\nOutput:\n{output_text.strip()}\n\nJSON:"
        inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
        
        trajectories = []
        prev_mode = model.training
        model.eval()
        

        for sample_idx in range(self.num_samples):
            try:
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=160,
                        do_sample=True,
                        temperature=self.sample_temperature,
                        top_p=0.9,
                        pad_token_id=tok.eos_token_id
                    )
                
                gen_text = tok.decode(gen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                logger.debug(f"ClarifyManager(student) sample {sample_idx}: {gen_text}")

                m = re.search(r'\{.*\}', gen_text, flags=re.S)
                if not m:
                    continue
                    
                try:
                    js = json.loads(m.group(0))
                    fmt_ok = True
                except Exception:
                    continue
                

                need = bool(js.get("need_clarify", False))
                if not need:
                    result = {"needs_clarify": False, "clarify_question": "", "issues": [], "type": ""}
                    q_len = 0
                else:
                    q = (js.get("clarify_question") or "").strip()
                    t = (js.get("type") or "").upper()
                    if t not in ("DG","RD","SC","CG"):
                        low_full = output_text.lower()
                        if any(k in low_full for k in ["missing","data","given","insufficient info"]):
                            t = "DG"
                        elif any(k in low_full for k in ["reasoning","logic","error"]):
                            t = "RD"
                        elif any(k in low_full for k in ["steps","detailed","explain"]):
                            t = "SC"
                        else:
                            t = "CG"
                    result = {"needs_clarify": True, "clarify_question": q[:200], "issues": [], "type": t}
                    try:
                        q_len = len(tok(q, add_special_tokens=False).input_ids) if q else 0
                    except Exception:
                        q_len = len(q)
                
                # Calculate trajectory log probability
                if self.online_rl:
                    model.train()
                    with torch.enable_grad():
                        full_ids = torch.cat([inputs['input_ids'], gen_ids[:, inputs['input_ids'].shape[1]:]], dim=1)
                        attention_mask = torch.ones_like(full_ids)
                        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        
                        output_start = inputs['input_ids'].shape[1]
                        output_logits = logits[:, output_start-1:-1, :]
                        target_ids = gen_ids[:, output_start:]
                        log_probs = F.log_softmax(output_logits, dim=-1)
                        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                        trajectory_log_prob = target_log_probs.sum()
                    model.eval()
                else:
                    trajectory_log_prob = torch.tensor(0.0)
                
                trajectory = {
                    "sample_idx": sample_idx,
                    "result": result,
                    "log_prob": trajectory_log_prob.detach() if isinstance(trajectory_log_prob, torch.Tensor) else trajectory_log_prob,
                    "fmt_ok": fmt_ok,
                    "q_len": q_len,
                    "raw_output": gen_text,
                    "input_ids": inputs['input_ids'].detach().clone(),
                    "output_ids": gen_ids[:, inputs['input_ids'].shape[1]:].detach().clone()
                }
                trajectories.append(trajectory)
                
            except Exception as e:
                logger.debug(f"Failed to generate trajectory {sample_idx}: {e}")
                continue

        if self.online_rl:
            model.train()
        else:
            if not prev_mode:
                model.eval()
        
        return trajectories

    def compute_group_relative_advantage(self, trajectories: List[Dict], terminal_reward: float = None) -> List[float]:
        if len(trajectories) <= 1:
            return [0.0] * len(trajectories)

        rewards = []
        for traj in trajectories:
            result = traj["result"]
            z_t = result["needs_clarify"]
            fmt_ok = traj["fmt_ok"]

            n_next = False
            
            r_edge = self.compute_edge_reward(z_t, n_next, fmt_ok)

            total_reward = r_edge
            if terminal_reward is not None:
                total_reward += self.lambda_R * terminal_reward
                
            rewards.append(total_reward)

        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]
        
        return advantages

    def store_edge_interaction_grpo(self, trajectories: List[Dict], advantages: List[float], meta: Dict = None):
        """
        Store GRPO multi-trajectory edge interaction information
        """
        if self.mode != 'student' or not trajectories:
            return 0.0

        best_idx = advantages.index(max(advantages)) if advantages else 0
        best_trajectory = trajectories[best_idx]

        for traj, advantage in zip(trajectories, advantages):
            edge_info = {
                "z_t": traj["result"]["needs_clarify"],
                "n_next": False, 
                "fmt_ok": traj["fmt_ok"],
                "r_edge": advantage, 
                "log_prob": traj["log_prob"],
                "q_len": traj["q_len"],
                "sample_idx": traj["sample_idx"],
                "is_best": (traj["sample_idx"] == best_trajectory["sample_idx"]),
                "input_ids": traj["input_ids"],
                "output_ids": traj["output_ids"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if meta:
                edge_info.update(meta)
                
            self._episode_edges.append(edge_info)
        
        logger.debug(f"Stored GRPO trajectories: {len(trajectories)}, best_idx: {best_idx}, best_advantage: {max(advantages):.4f}")
        
        return best_trajectory["result"]

    def e_grpo_update(self, terminal_reward: float = None, baseline: float = 0.0):
        if self.mode != 'student':
            return
            
        if not (self.online_rl and self._student_model is not None):
            return
            
        if len(self._episode_edges) == 0:
            return
        
        try:
            self._student_model.train()
            
            if self._rl_opt is None:
                trainable = self._select_trainable_params()
                self._rl_opt = torch.optim.AdamW(trainable, lr=self.rl_lr)

            if self._reference_model is None:
                self._reference_model = type(self._student_model).from_pretrained(
                    self.student_model_path
                ).to(next(self._student_model.parameters()).device)
                self._reference_model.eval()
            
            total_loss = 0.0
            num_updates = 0
            
            for edge_info in self._episode_edges:
                if not edge_info.get("z_t", False):  
                    continue
                
                input_ids = edge_info.get("input_ids")
                output_ids = edge_info.get("output_ids")
                if input_ids is None or output_ids is None:
                    continue
                
                current_log_prob = self._get_policy_logprobs(input_ids, output_ids, self._student_model)
                old_log_prob = edge_info.get("log_prob", torch.tensor(0.0))
                
                if not isinstance(current_log_prob, torch.Tensor):
                    current_log_prob = torch.tensor(current_log_prob, requires_grad=True)
                if not isinstance(old_log_prob, torch.Tensor):
                    old_log_prob = torch.tensor(old_log_prob)

                ratio = torch.exp(current_log_prob - old_log_prob.detach())

                A_grpo = torch.tensor(edge_info["r_edge"])

                surr1 = ratio * A_grpo
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_grpo
                local_loss = -torch.min(surr1, surr2)

                global_loss = torch.tensor(0.0)
                if terminal_reward is not None:
                    A_glob = torch.tensor(terminal_reward - baseline)
                    surr1_glob = ratio * A_glob
                    surr2_glob = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_glob
                    global_loss = -self.lambda_R * torch.min(surr1_glob, surr2_glob)

                kl_loss = torch.tensor(0.0)
                try:
                    if self._reference_model is not None:
                        ref_log_prob = self._get_policy_logprobs(input_ids, output_ids, self._reference_model)
                        kl_div = current_log_prob - ref_log_prob
                        kl_loss = self.beta * kl_div
                except Exception:
                    pass
                

                edge_loss = local_loss + global_loss + kl_loss
                total_loss += edge_loss
                num_updates += 1
            
            if total_loss != 0 and hasattr(total_loss, 'backward') and num_updates > 0:
                self._rl_opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._student_model.parameters(), 1.0)
                self._rl_opt.step()
                
                logger.debug(f"E-GRPO update: total_loss={float(total_loss):.4f}, "
                           f"num_updates={num_updates}, avg_loss={float(total_loss)/num_updates:.4f}")
            
        except Exception as e:
            logger.debug(f"E-GRPO update failed: {e}", exc_info=True)

        self._episode_edges.clear()

    def _student_clarify(self, output_text: str):
        self._init_student_model()
        if self._student_model is None:
            return None
        tok = self._student_tokenizer
        model = self._student_model
        instr = (
            "You are a result compliance and information sufficiency evaluator. Given an agent's output, determine if clarification is needed from it or upstream.\n"
            "Return only JSON with fields: need_clarify(bool), to_agent(str|null), type(str|DG|RD|SC|CG|null), clarify_question(str|null).\n"
            "Rules:\n"
            "1) If output is placeholder, error, very short(<20 chars) or lacks necessary values/conclusions, set need_clarify=True and provide a question <=25 words.\n"
            "2) Map type based on question: Data Gap=DG, Reasoning Defect=RD, Insufficient Steps=SC, Format/Answer Non-compliance=CG.\n"
            "3) If no clarification needed, need_clarify=false and other fields null.\n"
            "4) Strict JSON only, no extra text."
        )
        prompt = f"{instr}\n\nOutput:\n{output_text.strip()}\n\nJSON:"
        inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
        prev_mode = model.training
        model.eval()
        import torch as _torch
        with _torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=True,
                top_p=0.9
            )
        if self.online_rl:
            model.train()
        else:
            if not prev_mode:
                model.eval()
        try:
            self._last_inputs_ids = inputs['input_ids'].detach().clone()
            self._last_out_ids = gen_ids[:, inputs['input_ids'].shape[1]:].detach().clone()
        except Exception:
            self._last_inputs_ids, self._last_out_ids = None, None

        m = re.search(r'\{.*\}', gen_text, flags=re.S)
        if not m:
            self._last_fmt_ok = False
            self._last_q_len = 0
            return None
        try:
            js = json.loads(m.group(0))
            fmt_ok_local = True
        except Exception:
            self._last_fmt_ok = False
            self._last_q_len = 0
            return None
        need = bool(js.get("need_clarify", False))
        if not need:
            self._last_fmt_ok = True
            self._last_q_len = 0
            return {"needs_clarify": False, "clarify_question": "", "issues": [], "type": ""}
        q = (js.get("clarify_question") or "").strip()
        t = (js.get("type") or "").upper()
        if t not in ("DG","RD","SC","CG"):
            low_full = output_text.lower()
            if any(k in low_full for k in ["missing","data","given","insufficient info"]):
                t = "DG"
            elif any(k in low_full for k in ["reasoning","logic","error"]):
                t = "RD"
            elif any(k in low_full for k in ["steps","detailed","explain"]):
                t = "SC"
            else:
                t = "CG"

        self._last_fmt_ok = fmt_ok_local
        try:

            self._last_q_len = len(tok(q, add_special_tokens=False).input_ids) if q else 0
        except Exception:
            self._last_q_len = len(q)
        return {"needs_clarify": True, "clarify_question": q[:200], "issues": [], "type": t}

    def check(self, system_roles_desc: str, topology_desc: str, current_agent_role: str, next_agent_roles: list, output_text: str, role_prompt: str = ""):

        self._last_fmt_ok = False
        self._last_q_len = 0
        self._last_inputs_ids, self._last_out_ids = None, None
        if not self.enabled or self.mode == 'none':
            return {"needs_clarify": False, "clarify_question": ""}

        if self.mode == 'student':
            if self.online_rl and self.num_samples > 1:

                trajectories = self._student_clarify_multi_sample(output_text)
                if trajectories:

                    advantages = self.compute_group_relative_advantage(trajectories)
                    self._current_trajectories = list(zip(trajectories, advantages))

                    best_idx = advantages.index(max(advantages)) if advantages else 0
                    best_result = trajectories[best_idx]["result"]

                    self._last_fmt_ok = trajectories[best_idx]["fmt_ok"]
                    self._last_q_len = trajectories[best_idx]["q_len"]
                    
                    logger.debug(f"ClarifyManager: GRPO sampled {len(trajectories)} trajectories, best: {best_result}")
                    return best_result
                else:
                    logger.debug("ClarifyManager: GRPO sampling failed, fallback to single sample")

            stu_parsed = self._student_clarify(output_text)
            if stu_parsed is not None:
                logger.debug(f"ClarifyManager: student parsed -> {stu_parsed}")
                return stu_parsed
            logger.debug("ClarifyManager: student clarify failed, fallback heuristic.")
            return self.heuristic_check(output_text)

        tpl = self._load_template()
        system = tpl.get("system", "")
        user = tpl.get("user", "").format(
            system_roles_desc=system_roles_desc,
            topology_desc=topology_desc,
            current_agent_role=current_agent_role,
            next_agent_roles=(", ".join(next_agent_roles) if next_agent_roles else "None"),
            role_prompt=role_prompt,
            output_text=output_text
        )
        parsed = self._call_llm_raw(system, user)
        logger.debug(f"ClarifyManager: LLM raw response: {parsed}")
        try:
            normalized = self._normalize_parsed(parsed) if parsed is not None else None
            if normalized is not None:
                self._last_fmt_ok = True
                cq = normalized.get("clarify_question") or ""
                self._last_q_len = len(cq)
                if "type" not in normalized:
                    normalized["type"] = ""
                if "issues" not in normalized:
                    normalized["issues"] = []
                logger.debug(f"ClarifyManager: normalized: {normalized}")
                return normalized
            else:
                logger.debug("ClarifyManager: could not normalize LLM response, falling back to heuristic.")
        except Exception as e:
            logger.debug(f"ClarifyManager: normalization failed: {e}", exc_info=True)

        # Heuristic: not considered format correct
        self._last_fmt_ok = False
        self._last_q_len = 0
        return self.heuristic_check(output_text)

    def save_request(self, entry: dict):
        try:
            with self.output_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"ClarifyManager: Failed to save clarify request: {e}", exc_info=True)
            
            return None
        import torch
        model = self._student_model
        device = next(model.parameters()).device
        inp = self._last_inputs_ids.to(device)          # [1, L]
        out = self._last_out_ids.to(device)             # [1, M]
        full = torch.cat([inp, out], dim=1)             # [1, L+M]
        attn = torch.ones_like(full, dtype=torch.long)

        # Key: enable gradients and ensure model is in training mode (won't affect generation, only BN/Dropout behavior)
        model.train()  # Online training scenarios require training mode
        with torch.enable_grad():
            out_all = model(input_ids=full, attention_mask=attn)
            logits = out_all.logits  # [1, L+M, V]
            # Extract aligned prediction logits for generation segment
            pred_logits = logits[:, (full.shape[1]-out.shape[1]-1):-1, :]  # [1, M, V]
            target = out  # [1, M]
            logprobs = torch.log_softmax(pred_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, M]
            return logprobs.sum()  # scalar


    # New: Student model clarification reasoning
    def _student_clarify(self, output_text: str):
        """
        Unified output structure:
        {
          "needs_clarify": bool,
          "clarify_question": str,
          "issues": [],
          "type": ""
        }
        """
        self._init_student_model()
        if self._student_model is None:
            return None  # Fallback to heuristic
        import torch, re, json
        tok = self._student_tokenizer
        model = self._student_model
        instr = (
            "You are a result compliance and information sufficiency evaluator. Given an agent's output, determine if clarification is needed from it or upstream.\n"
            "Return only JSON with fields: need_clarify(bool), to_agent(str|null), type(str|DG|RD|SC|CG|null), clarify_question(str|null).\n"
            "Rules:\n"
            "1) If output is placeholder, error, very short(<20 chars) or lacks necessary values/conclusions, set need_clarify=True and provide a question <=25 words.\n"
            "2) Map type based on question: Data Gap=DG, Reasoning Defect=RD, Insufficient Steps=SC, Format/Answer Non-compliance=CG.\n"
            "3) If no clarification needed, need_clarify=false and other fields null.\n"
            "4) Strict JSON only, no extra text."
        )
        prompt = f"{instr}\n\nOutput:\n{output_text.strip()}\n\nJSON:"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        # Use eval + no_grad during generation for stability and speed
        prev_mode = model.training
        model.eval()
        import torch as _torch
        with _torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=True,
                top_p=0.9
            )
        # Restore to training mode (if online_rl) for subsequent _last_logprob/rl_update to generate gradients
        if self.online_rl:
            model.train()
        else:
            # Maintain eval if online training is disabled
            if not prev_mode:
                model.eval()

        gen_text = tok.decode(gen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        logger.debug(f"ClarifyManager(student) raw gen: {gen_text}")
        # Cache latest generation (for online RL, note only caching ids, actual logprob is "forward again" before rl_update)
        try:
            self._last_inputs_ids = inputs['input_ids'].detach().clone()
            self._last_out_ids = gen_ids[:, inputs['input_ids'].shape[1]:].detach().clone()
        except Exception:
            self._last_inputs_ids, self._last_out_ids = None, None

        m = re.search(r'\{.*\}', gen_text, flags=re.S)
        if not m:
            self._last_fmt_ok = False
            self._last_q_len = 0
            return None
        try:
            js = json.loads(m.group(0))
            fmt_ok_local = True
        except Exception:
            self._last_fmt_ok = False
            self._last_q_len = 0
            return None
        # Compatibility/normalization
        need = bool(js.get("need_clarify", False))
        if not need:
            self._last_fmt_ok = True
            self._last_q_len = 0
            return {"needs_clarify": False, "clarify_question": "", "issues": [], "type": ""}
        q = (js.get("clarify_question") or "").strip()
        t = (js.get("type") or "").upper()
        if t not in ("DG","RD","SC","CG"):
            low_full = output_text.lower()
            if any(k in low_full for k in ["missing","data","given","insufficient info"]):
                t = "DG"
            elif any(k in low_full for k in ["reasoning","logic","error"]):
                t = "RD"
            elif any(k in low_full for k in ["steps","detailed","explain"]):
                t = "SC"
            else:
                t = "CG"

        self._last_fmt_ok = fmt_ok_local
        try:

            self._last_q_len = len(tok(q, add_special_tokens=False).input_ids) if q else 0
        except Exception:
            self._last_q_len = len(q)
        return {"needs_clarify": True, "clarify_question": q[:200], "issues": [], "type": t}

    def check(self, system_roles_desc: str, topology_desc: str, current_agent_role: str, next_agent_roles: list, output_text: str, role_prompt: str = ""):

        self._last_fmt_ok = False
        self._last_q_len = 0
        self._last_inputs_ids, self._last_out_ids = None, None
        if not self.enabled or self.mode == 'none':
            return {"needs_clarify": False, "clarify_question": ""}

        if self.mode == 'student':
            if self.online_rl and self.num_samples > 1:

                trajectories = self._student_clarify_multi_sample(output_text)
                if trajectories:

                    advantages = self.compute_group_relative_advantage(trajectories)
                    self._current_trajectories = list(zip(trajectories, advantages))
 
                    best_idx = advantages.index(max(advantages)) if advantages else 0
                    best_result = trajectories[best_idx]["result"]

                    self._last_fmt_ok = trajectories[best_idx]["fmt_ok"]
                    self._last_q_len = trajectories[best_idx]["q_len"]
                    
                    logger.debug(f"ClarifyManager: GRPO sampled {len(trajectories)} trajectories, best: {best_result}")
                    return best_result
                else:
                    logger.debug("ClarifyManager: GRPO sampling failed, fallback to single sample")

            stu_parsed = self._student_clarify(output_text)
            if stu_parsed is not None:
                logger.debug(f"ClarifyManager: student parsed -> {stu_parsed}")
                return stu_parsed
            logger.debug("ClarifyManager: student clarify failed, fallback heuristic.")
            return self.heuristic_check(output_text)

        tpl = self._load_template()
        system = tpl.get("system", "")
        user = tpl.get("user", "").format(
            system_roles_desc=system_roles_desc,
            topology_desc=topology_desc,
            current_agent_role=current_agent_role,
            next_agent_roles=(", ".join(next_agent_roles) if next_agent_roles else "None"),
            role_prompt=role_prompt,
            output_text=output_text
        )
        parsed = self._call_llm_raw(system, user)
        logger.debug(f"ClarifyManager: LLM raw response: {parsed}")
        try:
            normalized = self._normalize_parsed(parsed) if parsed is not None else None
            if normalized is not None:

                self._last_fmt_ok = True
                cq = normalized.get("clarify_question") or ""
                self._last_q_len = len(cq)
                if "type" not in normalized:
                    normalized["type"] = ""
                if "issues" not in normalized:
                    normalized["issues"] = []
                logger.debug(f"ClarifyManager: normalized: {normalized}")
                return normalized
            else:
                logger.debug("ClarifyManager: could not normalize LLM response, falling back to heuristic.")
        except Exception as e:
            logger.debug(f"ClarifyManager: normalization failed: {e}", exc_info=True)

        self._last_fmt_ok = False
        self._last_q_len = 0
        return self.heuristic_check(output_text)
