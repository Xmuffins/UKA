from active_sampler import ActiveSampler
from knowledge_base import KnowledgeBaseChroma
import re
import math
import os
import requests
import time
import numpy as np

def call_local_llm(payload):
    base_url = os.getenv("LOCAL_LLM_CHAT_URL", "http://0.0.0.0:8000/v1/chat/completions")
    api_key = os.getenv("POLICY_API_KEY", "EMPTY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    retry = 10

    while retry > 0:
        print(retry)
        try:
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=360
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            retry -= 1
            if retry == 0:
                raise RuntimeError(f"LLM request failed after retries: {exc}") from exc
            time.sleep(2)

class Active_Responser:
    def __init__(self, policy_model):
        self.policy_model = './'+policy_model
        self.kb = KnowledgeBaseChroma(policy_model)


        scene_descriptions = [
            "讨论带有情绪色彩时，防御性反应容易加深误解。",
            "用户反馈表述不清晰或不满时，适合主动追问具体原因。",
            "用户强烈反应时，是提升论证精度的信号。",
            "当对话气氛紧张或争议激烈时，切换到中性话题以缓和情绪。",
            "争论激烈或分歧严重时，暂停并总结分歧点更有助于沟通。",
        ]

        documents = [
            "在有情绪的语境下，采取防御态度通常会加深误解。",
            "我可以请用户说明哪些地方让他觉得不清楚或感到沮丧。",
            "当有人强烈反应时，通常是我需要提升推理准确性的信号。",
            "我可以把话题转移到中性内容，以缓解情绪紧张。",
            "有时候，暂停下来总结争议的关键点是最好的选择。",
        ]

        metadata = [{'source':x} for x in documents]
        self.kb.add_documents(scene_descriptions,metadata)

        self.sampler = ActiveSampler(
            model=policy_model,
            KB=self.kb,
        )

        self.User_modeling_start_prompt = '''你是一位具有十年咨询经验的心理咨询师。请根据最新的用户对话，精炼地对用户的内心描述。
        历史对话：
        {}
        请不要回复分析、解释等内容，只用一句话简短给出新用户描述.
        '''


    def answering_strategy(self,history,user_model,is_training:bool=True):



        def user_model_update(user_model_info, query_history):
            '''
            user_model_info:
            {
            'user_model': list of user-profile hypotheses,
            "scores": scores,
            "probs": probs,
            "entropy": entropy
            }
            '''

            true_user_reply = query_history[-1]["content"]
            history_without_last = query_history[:-1]
            user_model = user_model_info['user_model']

            scores = []

            for z in user_model:
                messages = [
                    {
                        "role": "system",
                        "content": f"你将扮演用户‘user’，你现在的需求是：{z}"
                    }
                ]



                messages.append({"role": "user","content": true_user_reply})

                payload = {
                    "n": 1,
                    "max_tokens": 1,
                    "top_p": 1.0,
                    "temperature": 0,
                    "model": self.policy_model,
                    "messages": messages,
                    "logprobs": True,
                    "echo": True,
                }

                data = call_local_llm(payload)


                token_logprobs = [next(iter(d.values()))['logprob'] for d in data["prompt_logprobs"] if d is not None]


                valid_lps = [lp for lp in token_logprobs if lp is not None]
                avg_logprob = sum(valid_lps) / max(len(valid_lps), 1)

                scores.append(avg_logprob)

            if scores != []:
                max_score = max(scores)
                exp_scores = [math.exp(s - max_score) for s in scores]
                sum_exp = sum(exp_scores)
                probs = [e / sum_exp for e in exp_scores]


                entropy = -sum(
                    p * math.log(p + 1e-12) for p in probs
                )

            if user_model_info['scores'] == [] or max(scores) > max(user_model_info['scores']) or entropy > user_model_info['entropy']:

                if len(scores) == 2:
                    idx = probs.index(min(probs))
                    user_model.pop(idx)
                    scores.pop(idx)
                    probs.pop(idx)

                history_text = '\n'.join([f"{item['role']}: {item['content']}" for item in query_history[-3:]])
                user_model_str = "\n".join(f"{item}" for item in user_model)
                messages = [{'role':'user','content':self.User_modeling_start_prompt.format(history_text)}]

                payload = {
                    "n":1,
                    "max_tokens":4096,
                    "top_p":0.9,
                    "temperature":0.8,
                    "model": self.policy_model,
                    "messages": messages,
                    "logprobs": False
                }

                new_user_model = self.sampler.generate_candidates(payload)
                user_model.append(new_user_model[0])

                messages = [
                    {
                        "role": "system",
                        "content": f"{new_user_model}"
                    }
                ]



                messages.append({"role": "user","content": true_user_reply})

                payload = {
                    "n": 1,
                    "max_tokens": 1,
                    "top_p": 1.0,
                    "temperature": 0,
                    "model": self.policy_model,
                    "messages": messages,
                    "logprobs": True,
                    "echo": True,
                }

                data = call_local_llm(payload)


                token_logprobs = [next(iter(d.values()))['logprob'] for d in data["prompt_logprobs"] if d is not None]


                valid_lps = [lp for lp in token_logprobs if lp is not None]
                avg_logprob = sum(valid_lps) / max(len(valid_lps), 1)
                scores.append(avg_logprob)

                if scores != []:
                    max_score = max(scores)
                    exp_scores = [math.exp(s - max_score) for s in scores]
                    sum_exp = sum(exp_scores)
                    probs = [e / sum_exp for e in exp_scores]


                    entropy = -sum(
                        p * math.log(p + 1e-12) for p in probs
                    )

            return {
                "user_model": user_model,
                "scores": scores,
                "probs": probs,
                "entropy": entropy
            }



        def query_transfer_strategy(user_model,query_history):
            summary_prompt = """
            你是一位具有十年咨询经验的心理咨询师。请用一句话进行现状梳理，前半句话总结用户画像的关键点，后半句话总结用户的对话行为。请不要输出你的推理和中间过程，在思考后只输出总结部分。
            用户画像：
            {}
            历史对话：
            {}
            """

            user_model_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(user_model['user_model']))
            history_text = '\n'.join([f"{item['role']}: {item['content']}" for item in query_history[-3:]])

            message = [{'role':'user','content':summary_prompt.format(user_model_str,history_text)}]

            payload = {
                "n": 1,
                "max_tokens": 512,
                "top_p": 0.9,
                "temperature": 0.7,
                "model": self.policy_model,
                "messages": message,
                "logprobs": False
            }

            if self.policy_model == './seed-36b':
                payload['chat_template_kwargs'] = { "thinking_budget": 0 }

            solution_str = self.sampler.generate_candidates(payload)
            solution_str = solution_str[0].replace('总结','').strip()

            return solution_str

        def compute_tom_uncertainty(candidates, user_model, query_history):
            """
            Estimate candidate uncertainty by simulating replies under each
            user-profile hypothesis and measuring embedding dispersion.
            """

            uncertainties = []

            for c in candidates:
                embed_under_u = []

                for u in user_model:

                    messages = []
                    messages.append({
                        "role": "system",
                        "content": f"你将在下面的对话中扮演用户，你的角色设定为:\n{u}"
                    })
                    messages.extend(query_history)
                    messages.append({
                        "role": "assistant",
                        "content": c
                    })

                    payload = {
                        "n": 2,
                        "max_tokens": 2048,
                        "top_p": 0.9,
                        "temperature": 0.8,
                        "model": self.policy_model,
                        "messages": messages,
                        "logprobs": False
                    }

                    if self.policy_model == './seed-36b':
                        payload['chat_template_kwargs'] = { "thinking_budget": 0 }



                    group = self.sampler.generate_candidates(payload)

                    group_embeddings = self.kb.embedding_function.embedding_model.encode(group)
                    group_embedding = np.mean(group_embeddings, axis=0)









                    embed_under_u.append(group_embedding)




                E = np.stack(embed_under_u, axis=0)
                E = E / np.linalg.norm(E, axis=1, keepdims=True)

                centroid = E.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)

                cosine_sims = E @ centroid
                uncertainty = 1.0 - cosine_sims.mean()

                uncertainties.append(uncertainty)

            return uncertainties







        user_model = user_model_update(user_model,history)


        transfered_query = query_transfer_strategy(user_model,history)


        kb_ctx = self.sampler.KB.retrieve_topk(transfered_query,5)
        kb_ctx_str = '\n'.join([content['metadata']['source'] for content in kb_ctx])


        prompt = (
                f"你是一位具有十年咨询经验的心理咨询师。在下面的对话中，你将尝试与一位用户进行对话，并满足用户的心理需求。请基于下面的对话场景，为用户提供一个高情商且有效满足用户需求的回复。\n经验:\n{kb_ctx_str}\n"
            )

        messages = [{"role": "system", "content": prompt}] + history

        payload = {
            "n":4,
            "max_tokens":4096,
            "top_p":0.9,
            "temperature":0.8,
            "model": self.policy_model,
            "messages": messages,
            "logprobs": False
            }




        candidates = self.sampler.generate_candidates(payload)



        candidates_probs = compute_tom_uncertainty(candidates,user_model,history)


        scores = self.sampler.information_gain(candidates,candidates_probs)


        if is_training:


            chosen_candidate = self.sampler.choose_candidate(scores,None)
        else:

            chosen_candidate = self.sampler.choose_candidate(scores,'greedy')




        return transfered_query,chosen_candidate,user_model





    def updating_memory_strategy(self,history,transfered_query):
        """
        Generate a structured memory entry from the user's latest feedback.
        """

        def judge_user_mood(query_history):
            judge_prompt = """
            请判断一位对话者在两次交流之间情绪发生了正向还是负向的变化，请用"<正向>"或"<负向>"给出你的最终判断。如果你拿不准，请回复"<不确定>"
            上次对话：
            {}
            本次对话：
            {}
            请只回复"<正向>"或"<负向>"。
            """
            payload = {
                "n":1,
                "max_tokens":2048,
                "top_p":0.9,
                "temperature":0.8,
                "model":self.policy_model,
                "messages": [{'role':'user','content':judge_prompt.format(query_history[-3]['content'],query_history[-1]['content'])}],
                "logprobs": False
                }
            if self.policy_model == './seed-36b':
                payload['chat_template_kwargs'] = { "thinking_budget": 512 }
            solution_str = self.sampler.generate_candidates(payload)


            solution = re.search(r"<(.*?)>", solution_str[0])
            if solution is None:
                final_answer = "不确定"
            else:
                final_answer = solution.group(0)

            return final_answer


        mood = judge_user_mood(history)


        if "不确定" in mood:
            print("no experience due to unpredictable mood change")
            return None


        last_assistant_response = '\n'.join([f"{item['role']}: {item['content']}" for item in history])



        interpret_prompt = f"""
            你是一名对话策略分析师。请基于对话，分析user的反馈，总结在此类情境下作为assistant应使用或避免的沟通方式，或者应如何有效的消除误解、提升对用户对话目标的理解。请直接给出经验内容，长度为一句话。

            对话历史：
            {last_assistant_response}

            用户情绪变化：
            {mood}

            请直接给出经验内容。
            """

        messages = [{'role':'user','content':interpret_prompt}]

        payload = {
            "n":1,
            "max_tokens":4096,
            "top_p":0.9,
            "temperature":0.8,
            "model":self.policy_model,
            "messages": messages,
            "logprobs": False
            }




        raw_output = self.sampler.generate_candidates(payload)
        raw_output = raw_output[0].replace('经验内容','').strip()
        print(f'\n\n\n\nExperience content: {raw_output}')



        entry = {'source':raw_output}


        self.kb.add_documents([transfered_query],[entry])
        print("currently we have knowledges ", str(len(self.kb.collection.get()['ids'])))
        print("currently we have knowledges another ", str(len(self.kb.collection_known.get()['ids'])))
        return entry















