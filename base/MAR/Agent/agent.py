from typing import Dict
import json
from loguru import logger
from datetime import datetime
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_registry import LLMRegistry
from MAR.Roles.role_registry import RoleRegistry
from MAR.Graph.node import Node
from MAR.Prompts.message_aggregation import message_aggregation,inner_test
from MAR.Prompts.post_process import post_process
from MAR.Prompts.output_format import output_format_prompt
from MAR.Prompts.reasoning import reasoning_prompt


@AgentRegistry.register('Agent')
class Agent(Node):
    def __init__(self, id: str | None =None, domain: str = "", role:str = None , llm_name: str = "",reason_name: str = "",):
        super().__init__(id, reason_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.role = RoleRegistry(domain, role)
        self.reason = reason_name

        self.message_aggregation = self.role.get_message_aggregation()
        self.description = self.role.get_description()
        self.output_format = self.role.get_output_format()
        self.post_process = self.role.get_post_process()
        self.post_description = self.role.get_post_description()
        self.post_output_format = self.role.get_post_output_format()
        # Reflect
        if reason_name == "Reflection" and self.post_output_format == "None":
            self.post_output_format = self.output_format
            self.post_description = "\nReflect on possible errors in the answer above and answer again using the same format. If you think there are no errors in your previous answers that will affect the results, there is no need to correct them.\n"


    @torch.no_grad()
    def _student_generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        try:
            tok = AutoTokenizer.from_pretrained(self._student_model_path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(self._student_model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device).eval()
            combined = f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n输出:"
            inputs = tok(combined, return_tensors="pt").to(device)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9
            )
            gen_text = tok.decode(gen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            logger.info(f"STUDENT_MODEL_RAW: {gen_text}")
            return gen_text.strip()
        except Exception as e:
            logger.error(f"[StudentModel] immediate load failed: {e}")
            return "[StudentModelError] Fallback: model not available."

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str, Dict], temporal_info:Dict[str, Dict], **kwargs):
        query = raw_inputs['query']
        spatial_prompt = message_aggregation(raw_inputs, spatial_info, self.message_aggregation)
        temporal_prompt = message_aggregation(raw_inputs, temporal_info, self.message_aggregation)
        format_prompt = output_format_prompt[self.output_format]
        reason_prompt = reasoning_prompt[self.reason]

        system_prompt = f"{self.description}\n{reason_prompt}"
        system_prompt += f"\nFormat requirements that must be followed:\n{format_prompt}" if format_prompt else ""
        user_prompt = f"{query}\n"
        user_prompt += f"At the same time, other agents' outputs are as follows:\n\n{spatial_prompt}" if spatial_prompt else ""
        user_prompt += f"\n\nIn the last round of dialogue, other agents' outputs were:\n\n{temporal_prompt}" if temporal_prompt else ""
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        query = input['query']
        passed, response= inner_test(input, spatial_info, temporal_info)
        if passed:
            return response
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)

        response = self.llm.gen(prompt)

        response = post_process(input, response, self.post_process)
        logger.debug(f"Agent {self.id} Role: {self.role.role} LLM: {self.llm.model_name}")
        logger.debug(f"system prompt:\n {prompt[0]['content']}")
        logger.debug(f"user prompt:\n {prompt[1]['content']}")
        logger.debug(f"response:\n {response}")


        try:
            clarify_mgr = None
            if hasattr(self, 'graph') and hasattr(self.graph, 'clarify_manager'):
                clarify_mgr = getattr(self.graph, 'clarify_manager')
                
            if clarify_mgr is not None and getattr(clarify_mgr, 'enabled', False):
                system_roles_desc = str(self.graph.list_nodes()) if hasattr(self.graph, 'list_nodes') else ""
                topology_desc = ""
                if hasattr(self.graph, 'spatial_adj_matrix'):
                    topology_desc += f"spatial_adj: {getattr(self.graph, 'spatial_adj_matrix')}\n"
                if hasattr(self.graph, 'temporal_adj_matrix'):
                    topology_desc += f"temporal_adj: {getattr(self.graph, 'temporal_adj_matrix')}\n"
                current_agent_role = getattr(self.role, 'role', str(self.role))
                next_agent_roles = [s.role.role for s in self.spatial_successors] if hasattr(self, 'spatial_successors') else []
                output_text = str(response)
                role_prompt = prompt[0]['content'] if isinstance(prompt, list) and len(prompt)>0 else getattr(self, 'description', "")
                parsed = clarify_mgr.check(system_roles_desc, topology_desc, current_agent_role, next_agent_roles, output_text, role_prompt=role_prompt)
                logger.debug(f"Clarify check parsed: {parsed}")
                fmt_ok_flag = bool(getattr(clarify_mgr, "_last_fmt_ok", False))
                q_len_flag = int(getattr(clarify_mgr, "_last_q_len", 0))

                if isinstance(parsed, dict) and parsed.get("needs_clarify", False):
                    req_id = clarify_mgr.new_request_id() if hasattr(clarify_mgr, 'new_request_id') else f"clarify_{uuid.uuid4().hex[:6]}"
                    if len(next_agent_roles) > 0:
                        to_agent = next_agent_roles[0]
                    else:
                        to_agent = parsed.get("ask_to") or current_agent_role
                    if parsed.get("needs_clarify", False):
                        entry = {
                            "id": req_id,
                            "task": getattr(self.graph, 'domain', "") or "",
                            "from_agent": current_agent_role,
                            "to_agent": to_agent,
                            "query": input.get('query') if isinstance(input, dict) else None,
                            "original_response": output_text,
                            "new_response": None,
                            "agent_prompt": role_prompt,
                            "reasoning": getattr(self, 'reason', "") or getattr(self.graph, 'reasoning_name', ""),
                            "system_roles": system_roles_desc,
                            "clarify_question": parsed.get("clarify_question", ""),
                            "needs_clarify": True,
                            "ask_reason": parsed.get("ask_reason", ""),
                            "issues": parsed.get("issues", []),
                            "confidence": parsed.get("confidence", None),
                            "type": parsed.get("type", "") ,
                            "timestamp": datetime.utcnow().isoformat() + "Z"
                        }
                    else:
                        entry = {
                            "id": req_id,
                            "task": getattr(self.graph, 'domain', "") or "",
                            "from_agent": current_agent_role,
                            "to_agent": to_agent,
                            "query": input.get('query') if isinstance(input, dict) else None,
                            "original_response": output_text,
                            "new_response": None,
                            "needs_clarify": False,
                            "agent_prompt": role_prompt,
                            "reasoning": getattr(self, 'reason', "") or getattr(self.graph, 'reasoning_name', ""),
                            "system_roles": system_roles_desc,
                            "clarify_question": parsed.get("clarify_question", ""),
                            "ask_reason": parsed.get("ask_reason", ""),
                            "issues": parsed.get("issues", []),
                            "confidence": parsed.get("confidence", None),
                            "type": parsed.get("type", "") ,
                            "timestamp": datetime.utcnow().isoformat() + "Z"
                        }
                    logger.info(f"CLARIFY_REQUEST_ID:{req_id} -> {entry['clarify_question']}")
                    try:
                        follow_system = prompt[0]['content'] if isinstance(prompt, list) and len(prompt)>0 else role_prompt
                        follow_user = (f"{query}\n"
                                       f"Initial thinking information:\n{output_text}\n\n"
                                       f"Follow-up question: {entry['clarify_question']}\n"
                                       "Please answer concisely. If you change any intermediate step, state it explicitly.")
                        follow_prompt = [{'role': 'system', 'content': follow_system}, {'role': 'user', 'content': follow_user}]
                        logger.debug(f"Clarify followup system prompt:\n {follow_system}")
                        logger.debug(f"Clarify followup user prompt:\n {follow_user}")
                        clarify_response = self.llm.gen(follow_prompt)
                        logger.debug(f"Clarify followup response:\n {clarify_response}")
                        combined_response = f"{response}\n\n--- CLARIFICATION ({req_id}) ---\n{clarify_response}"
                        response = combined_response
                        entry["new_response"] = clarify_response
                        entry["combined_response"] = combined_response
                        entry["saved_at"] = datetime.utcnow().isoformat() + "Z"
                        if "type" not in entry:
                            entry["type"] = parsed.get("type", "")
                        try:
                            clarify_mgr.save_request(entry)
                        except Exception:
                            logger.debug("Failed to save clarify entry", exc_info=True)
                        try:
                            nxt = clarify_mgr.check(system_roles_desc, topology_desc, current_agent_role, next_agent_roles, combined_response, role_prompt=role_prompt)
                            n_next = bool(nxt.get("needs_clarify", False))
                            if clarify_mgr.mode == 'student' and hasattr(clarify_mgr, '_current_trajectories') and clarify_mgr._current_trajectories:
                                trajectories, advantages = zip(*clarify_mgr._current_trajectories)

                                updated_trajectories = []
                                updated_advantages = []
                                for traj, adv in zip(trajectories, advantages):
                                    z_t = traj["result"]["needs_clarify"]
                                    fmt_ok = traj["fmt_ok"]
                                    r_edge_updated = clarify_mgr.compute_edge_reward(z_t, n_next, fmt_ok)
                                    
                                    traj_updated = traj.copy()
                                    traj_updated["n_next"] = n_next
                                    traj_updated["r_edge_updated"] = r_edge_updated
                                    
                                    updated_trajectories.append(traj_updated)
                                    updated_advantages.append(adv)  
                                
                                result = clarify_mgr.store_edge_interaction_grpo(
                                    updated_trajectories,
                                    updated_advantages,
                                    meta={
                                        "agent_id": self.id,
                                        "agent_role": current_agent_role,
                                        "clarify_question": entry.get("clarify_question", "")[:100]
                                    }
                                )
                                
                                best_advantage = max(updated_advantages) if updated_advantages else 0.0
                                logger.info(f"RL_GRPO_REWARD agent={self.id} role={current_agent_role} -> "
                                          f"z_t=1 n_next={int(n_next)} trajectories={len(updated_trajectories)} best_adv={best_advantage:.4f}")

                                clarify_mgr._current_trajectories = []
                                
                            else:
                                current_log_prob = torch.tensor(0.0)
                                if clarify_mgr.mode == 'student' and hasattr(clarify_mgr, '_last_logprob'):
                                    try:
                                        current_log_prob = clarify_mgr._last_logprob()
                                    except Exception:
                                        pass
                                r_edge = clarify_mgr.store_edge_interaction(
                                    z_t=True,
                                    n_next=n_next,
                                    fmt_ok=fmt_ok_flag,
                                    log_prob=current_log_prob,
                                    q_len=q_len_flag,
                                    meta={
                                        "agent_id": self.id,
                                        "agent_role": current_agent_role,
                                        "clarify_question": entry.get("clarify_question", "")[:100]
                                    }
                                )

                                if clarify_mgr.mode == 'student':
                                    logger.info(f"RL_EDGE_REWARD agent={self.id} role={current_agent_role} -> "
                                              f"z_t=1 n_next={int(n_next)} fmt_ok={int(fmt_ok_flag)} q_len={q_len_flag} r_edge={r_edge:.4f}")
                                else:
                                    logger.debug(f"LLM_MODE: No reward calculation for agent={self.id} role={current_agent_role}")
                            
                        except Exception:
                            logger.debug("E-GRPO edge reward recording failed.", exc_info=True)

                    except Exception as e:
                        logger.error(f"Follow-up clarify failed for Agent {self.id}: {e}", exc_info=True)
                        try:
                            entry["saved_at"] = datetime.utcnow().isoformat() + "Z"
                            if "type" not in entry:
                                entry["type"] = parsed.get("type", "")
                            clarify_mgr.save_request(entry)
                        except Exception:
                            logger.debug("Failed to save fallback clarify entry", exc_info=True)
                else:
                    try:
                        if clarify_mgr.mode == 'student':
                            clarify_mgr.store_edge_interaction(
                                z_t=False,
                                n_next=False,
                                fmt_ok=False,
                                log_prob=torch.tensor(0.0),
                                q_len=0,
                                meta={"agent_id": self.id, "agent_role": current_agent_role}
                            )
                            logger.info(f"RL_EDGE_REWARD agent={self.id} role={current_agent_role} -> z_t=0 r_edge=0.0")
                        else:
                            logger.debug(f"LLM_MODE: No reward recording for non-clarify interaction, agent={self.id}")
                    except Exception:
                        logger.debug("E-GRPO non-clarify recording failed.", exc_info=True)
                        
            else:
                 logger.debug(f"No clarify manager or not enabled for Agent {self.id} Role: {self.role.role}")
        except Exception:
            logger.error(f"Error during clarify check in Agent {self.id} Role: {self.role.role}", exc_info=True)
            pass

        # #! 
        # received_id = []
        # for id, info in spatial_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')
        # for id, info in temporal_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')

        # entry = {
        #     "id": self.id,
        #     "role": self.role.role,
        #     "llm_name": self.llm.model_name,
        #     "system_prompt": prompt[0]['content'],
        #     "user_prompt": prompt[1]['content'],
        #     "received_id": received_id,
        #     "response": response,
        # }
        # try:
        #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     data = []

        # data.append(entry)

        # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
        # #!

        post_format_prompt = output_format_prompt[self.post_output_format]
        if post_format_prompt is not None:
            system_prompt = f"{self.post_description}\n"
            system_prompt += f"Format requirements that must be followed:\n{post_format_prompt}"
            user_prompt = f"{query}\nThe initial thinking information is:\n{response} \n Please refer to the new format requirements when replying."
            prompt = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
            response = self.llm.gen(prompt)
            logger.debug(f"post system prompt:\n {system_prompt}")
            logger.debug(f"post user prompt:\n {user_prompt}")
            logger.debug(f"post response:\n {response}")
        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None

@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None =None, agent_name = "", domain = "", llm_name = "", prompt_file = ""):
        super().__init__(id, agent_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_file = json.load(open(f"{prompt_file}", 'r', encoding='utf-8'))

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):  
        system_prompt = f"{self.prompt_file['system']}"
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"
        user_prompt = f"The task is:\n\n {raw_inputs['query']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str} {self.prompt_file['user']}"
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
    
    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        response = self.llm.gen(prompt)
        logger.debug(f"Final Refer Node LLM: {self.llm.model_name}")
        logger.debug(f"Final System Prompt:\n {prompt[0]['content']}")
        logger.debug(f"Final User Prompt:\n {prompt[1]['content']}")
        logger.debug(f"Final Response:\n {response}")
        # #! 
        # received_id = []
        # for id, info in spatial_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')
        # for id, info in temporal_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')

        # entry = {
        #     "id": self.id,
        #     "role": "FinalDecision",
        #     "llm_name": self.llm.model_name,
        #     "system_prompt": prompt[0]['content'],
        #     "user_prompt": prompt[1]['content'],
        #     "received_id": received_id,
        #     "response": response,
        # }
        # try:
        #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     data = []

        # data.append(entry)

        # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
        # #!
        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None