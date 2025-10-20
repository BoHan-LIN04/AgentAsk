import json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .templates import build_input, target_json_str
import re

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    ap.add_argument("--data", required=True, help="Path to evaluation data")
    ap.add_argument("--batch_size", type=int, default=8)
    return ap.parse_args()

def extract_json_from_output(text):
    try:
        pattern = r'\{[^}]*"need_clarify"[^}]*\}'
        match = re.search(pattern, text)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except:
        pass
    return {
        "need_clarify": False,
        "to_agent": None,
        "type": None,
        "clarify_question": None
    }

def evaluate_model(model, tokenizer, eval_data, batch_size=8):
    model.eval()
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i+batch_size]
            
            for example in batch:

                prompt = build_input(example)
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=1024, 
                    truncation=True
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[-1]:], 
                    skip_special_tokens=True
                )
                
                pred_json = extract_json_from_output(generated_text)
                predictions.append(pred_json)
                
                if 'target_json' in example:
                    ground_truths.append(example['target_json'])
                else:
                    gt = {
                        "need_clarify": bool(example.get('clarify_question')),
                        "to_agent": example.get('to_agent'),
                        "type": example.get('type'),
                        "clarify_question": example.get('clarify_question')
                    }
                    ground_truths.append(gt)
    
    return predictions, ground_truths

def compute_metrics(predictions, ground_truths):
    metrics = {}    
    need_pred = [p.get('need_clarify', False) for p in predictions]
    need_gt = [gt.get('need_clarify', False) for gt in ground_truths]
    
    metrics['need_clarify_accuracy'] = accuracy_score(need_gt, need_pred)
    metrics['need_clarify_f1'] = f1_score(need_gt, need_pred, average='weighted')
    
    clarify_indices = [i for i, gt in enumerate(ground_truths) if gt.get('need_clarify')]
    
    if clarify_indices:
        agent_pred = [predictions[i].get('to_agent') for i in clarify_indices]
        agent_gt = [ground_truths[i].get('to_agent') for i in clarify_indices]
        metrics['to_agent_accuracy'] = accuracy_score(agent_gt, agent_pred)
        
        type_pred = [predictions[i].get('type') for i in clarify_indices]
        type_gt = [ground_truths[i].get('type') for i in clarify_indices]
        metrics['type_accuracy'] = accuracy_score(type_gt, type_pred)
        
        exact_match = 0
        for i in clarify_indices:
            pred = predictions[i]
            gt = ground_truths[i]
            if (pred.get('need_clarify') == gt.get('need_clarify') and
                pred.get('to_agent') == gt.get('to_agent') and
                pred.get('type') == gt.get('type')):
                exact_match += 1
        
        metrics['exact_match_accuracy'] = exact_match / len(clarify_indices)
    
    return metrics

def main():
    args = parse()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    try:
        model = PeftModel.from_pretrained(base_model, args.model_path)
    except:
        model = base_model

    eval_data = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))
    
    print(f"Loaded {len(eval_data)} evaluation examples")
    

    predictions, ground_truths = evaluate_model(
        model, tokenizer, eval_data, args.batch_size
    )
    

    metrics = compute_metrics(predictions, ground_truths)
    

    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    

    results = {
        'metrics': metrics,
        'predictions': predictions[:10],  
        'ground_truths': ground_truths[:10]
    }
    
    output_path = f"{args.model_path}/eval_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
