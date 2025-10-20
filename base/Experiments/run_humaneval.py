import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import argparse
import yaml
import json
import time
import re
import torch
from loguru import logger
import torch.nn.functional as F

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Tools.reader.readers import JSONLReader
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Utils.utils import fix_random_seed, split_list
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging
from pathlib import Path
import uuid
from datetime import datetime
from MAR.clarify_manager import ClarifyManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on humaneval")
    parser.add_argument("--dataset_json", type=str, default="Datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16,help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--prompt_file', type=str, default='MAR/Roles/FinalNode/humaneval.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=200.0)
    parser.add_argument('--max_agent', type=int, default=6)
    parser.add_argument('--error-mode', type=str, default='none', choices=['none','realtime','postindex'])
    parser.add_argument('--error-dir', type=str, default='error_cases')
    parser.add_argument('--window-lines', type=int, default=10)
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--clarify', type=str, default='llm', choices=['none','llm','student'])
    parser.add_argument('--clarify-model', type=str, default='gpt-5')
    parser.add_argument('--clarify-student-path', type=str, default='Clarify/model.pt')
    parser.add_argument('--clarify-output', type=str, default='Clarify/humaneval.jsonl')
    

    parser.add_argument('--rl-alpha-eff', type=float, default=1.0, help='Effectiveness reward weight')
    parser.add_argument('--rl-alpha-fmt', type=float, default=0.5, help='Format reward weight')  
    parser.add_argument('--rl-alpha-ans', type=float, default=2.0, help='Terminal answer reward weight')
    parser.add_argument('--rl-lambda-sw', type=float, default=0.2, help='Sliding window penalty coefficient')
    parser.add_argument('--rl-lambda-R', type=float, default=1.0, help='Global reward weight')
    parser.add_argument('--rl-beta', type=float, default=0.01, help='KL regularization coefficient')
    parser.add_argument('--rl-epsilon', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--rl-H', type=int, default=5, help='Sliding window size')

    parser.add_argument('--grpo-num-samples', type=int, default=4, help='Number of trajectories to sample for GRPO')
    parser.add_argument('--grpo-temperature', type=float, default=1.0, help='Sampling temperature for GRPO')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = JSONLReader().parse_file("Datasets/humaneval/humaneval-py.jsonl")
    train_dataset, test_dataset = split_list(dataset, 0.2)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"humaneval_{current_time}.log"
    fix_random_seed(1234)
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter(max_agent=args.max_agent,device=device).to(device)
    clarify_enabled = (args.clarify != 'none')
    router.clarify_manager = ClarifyManager(
        enabled=clarify_enabled, 
        mode=args.clarify, 
        model=args.clarify_model, 
        student_model_path=args.clarify_student_path, 
        output_path=args.clarify_output, 
        task='humaneval', 
        prompt_dir='MAR/ClarifyPrompts'
    )

    if clarify_enabled and router.clarify_manager:
        router.clarify_manager.rl_config(
            alpha_eff=args.rl_alpha_eff,
            alpha_fmt=args.rl_alpha_fmt,
            alpha_ans=args.rl_alpha_ans,
            lambda_sw=args.rl_lambda_sw,
            lambda_R=args.rl_lambda_R,
            beta=args.rl_beta,
            epsilon=args.rl_epsilon,
            H=args.rl_H,
            num_samples=args.grpo_num_samples,
            sample_temperature=args.grpo_temperature
        )
        
  
        if router.clarify_manager.mode == 'student':
            router.clarify_manager.online_rl = True
            logger.info(f"Enabled online RL with GRPO: num_samples={args.grpo_num_samples}, temperature={args.grpo_temperature}")
    
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile


    error_out = Path(args.error_dir)
    error_out.mkdir(parents=True, exist_ok=True)
    index_path = error_out / "errors_index.jsonl"
    def new_case_id():
        return f"case_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    def append_index(case_id, info_dict):
        entry = {"id": case_id, **info_dict}
        with index_path.open('a', encoding='utf-8') as jf:
            jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    def write_realtime_case(case_id, info_dict):
        case_dir = error_out / case_id
        case_dir.mkdir(exist_ok=True)
        with (case_dir / "metadata.json").open('w', encoding='utf-8') as mf:
            json.dump(info_dict, mf, ensure_ascii=False, indent=2)
        with (case_dir / "snippet.log").open('w', encoding='utf-8') as sf:
            sf.write(info_dict.get("log_text", ""))
        return str(case_dir.resolve())
    def postprocess_index_and_log(index_file: Path, log_file_path: Path, out_dir: Path, window_lines: int = 10):
        if not index_file.exists() or not log_file_path.exists():
            logger.error("Index file or log file not found for postprocess.")
            return
        with log_file_path.open('r', encoding='utf-8', errors='replace') as lf:
            log_lines = lf.readlines()
        for line in index_file.open('r', encoding='utf-8'):
            try:
                entry = json.loads(line)
            except Exception:
                continue
            cid = entry.get("id")
            marker = f"ERROR_CASE_ID:{cid}"
            found_idx = None
            for i, l in enumerate(log_lines):
                if marker in l:
                    found_idx = i
                    break
            case_dir = out_dir / cid
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / "metadata.json").open('w', encoding='utf-8') as mf:
                json.dump(entry, mf, ensure_ascii=False, indent=2)
            if found_idx is not None:
                start = max(0, found_idx - window_lines)
                end = min(len(log_lines), found_idx + window_lines + 1)
                snippet = ''.join(log_lines[start:end])
            else:
                snippet = entry.get("log_text","")
            with (case_dir / "snippet.log").open('w', encoding='utf-8') as sf:
                sf.write(snippet)

    logger.info("Start training...")
    for epoch in range(args.epochs):
        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"humaneval_router_epoch{epoch}.pth", map_location=torch.device('cuda')))
            continue
        logger.info(f"Epoch {epoch}",80*'-')
        train_batches = int(len(train_dataset)/args.batch_size)
        total_solved, total_executed = (0, 0)
        for i_batch in range(train_batches):
            logger.info(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            current_batch = dataloader(train_dataset,args.batch_size,i_batch)
            queries = [item['prompt'] for item in current_batch]
            tests = [item['test'] for item in current_batch]
            task_labels = [2 for _ in current_batch]
            tasks_y = torch.tensor(task_labels).to(device)
            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)

            task_loss = F.cross_entropy(tasks_probs, tasks_y)
            utilities = []
            answers_loss = []
            is_solved_list = []
            pattern = r'```python.*```'
            for query, result, test, log_prob, cost in zip(queries, results, tests, log_probs, costs):
                match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                else:
                    answer = ""
                    is_solved = 0
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = is_solved - cost * args.cost_rate
                utilities.append(utility)
                is_solved_list.append(is_solved)
                answer_loss = -log_prob * utility
                answers_loss.append(answer_loss)
                if not is_solved:
                    cid = new_case_id()
                    log_text = (f"[{cid}] Query: {query}\n"
                                f"Cost: {cost}, LogProb: {log_prob}\n"
                                f"Utility: {utility}\n")
                    logger.info(f"ERROR_CASE_ID:{cid} {log_text}")
                    info = {
                        "id": cid,
                        "query": query,
                        "cost": float(cost) if hasattr(cost, 'item') else cost,
                        "log_prob": float(log_prob) if hasattr(log_prob, 'item') else log_prob,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "log_file": log_file,
                        "log_text": log_text
                    }
                    if args.error_mode == 'realtime':
                        write_realtime_case(cid, info)
                    elif args.error_mode == 'postindex':
                        append_index(cid, info)

            answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
            vae_loss = vae_loss.mean()
            is_solved_tensor = torch.tensor(is_solved_list, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [N, 1]
            adjust_loss = ((1 - is_solved_tensor) * (router.num_determiner.max_agent - agents_num) + 0.25 * is_solved_tensor *  agents_num).mean()
            
            loss = task_loss + answer_loss + vae_loss*0.001 # + adjust_loss
            loss.backward()
            optimizer.step()
        
            accuracy = total_solved / total_executed
            logger.info(f"Batch time {time.time() - start_ts:.3f}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"utilities:{utilities}")
        logger.info(f"Epoch {epoch} Finishes",80*'-')
        torch.save(router.state_dict(), f"humaneval_router_epoch{epoch}.pth")
    logger.info("Finish training...")
    logger.info("Start testing...")
    test_batches = int(len(test_dataset)/args.batch_size)
    total_solved, total_executed = (0, 0)
    
    for i_batch in range(test_batches):
        logger.info(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        current_batch = dataloader(test_dataset,args.batch_size,i_batch)
        queries = [item['prompt'] for item in current_batch]
        tests = [item['test'] for item in current_batch]
        task_labels = [2 for _ in current_batch]
        tasks_y = torch.tensor(task_labels).to(device)
        results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)

        utilities = []
        pattern = r'```python.*```'
        for query, result, test, log_prob, cost in zip(queries, results, tests, log_probs, costs):
            match = re.search(pattern, result, re.DOTALL|re.MULTILINE)
            if match:
                answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            else:
                is_solved = 0
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            utility = is_solved - cost * args.cost_rate
            utilities.append(utility)
    
        accuracy = total_solved / total_executed
        logger.info(f"Batch time {time.time() - start_ts:.3f}")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"utilities:{utilities}")
    logger.info("Finish testing...")
    if args.postprocess or args.error_mode == 'postindex':
        postprocess_index_and_log(index_path, Path(log_file), error_out, window_lines=args.window_lines)