from active_sampler import ActiveSampler
from knowledge_base import KnowledgeBaseChroma
import re
import math
import os
import requests
import time
from copy import deepcopy
import numpy as np

def call_local_llm(payload):

    if payload['model'] not in ['./qwen-32b','./gpt-120b','./seed-36b','./qwen3-235b-a22b-instruct-2507']:
        payload['model'] = payload['model'].replace('./','')
        base_url = os.getenv("REMOTE_LLM_CHAT_URL", "http://0.0.0.0:8000/v1/chat/completions")
    elif payload['model'] in ['./qwen-32b','./seed-36b']:
        base_url = os.getenv("LOCAL_LLM_CHAT_URL", "http://0.0.0.0:8000/v1/chat/completions")
    else:
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
    def __init__(self, policy_model,mode):
        self.Principle_revision_prompt = """You are a strategy refinement assistant.

        Your task is to suggest an improved strategy for the assistant to retry the current turn, based on previous failed attempts.

        [Task Goal]
        {task_goal}

        [Dialogue History Before Current Turn]
        {dialogue_history}

        [Previous Failed Trials]
        {failed_trials}

        Instructions:
        1. Identify why the previous attempts failed to improve the user's emotional state or task progress.
        2. Propose a different high-level strategy for the assistant to try at the same dialogue state.
        3. Do not generate the final assistant reply.
        4. Output one concise strategy sentence.

        Output format:
        Rationale:
        [brief rationale]

        Improved Strategy:
        [one sentence strategy]
        """
        self.policy_model = './'+policy_model
        self.kb = KnowledgeBaseChroma(policy_model+'_'+mode+'_esconv')

        scene_descriptions = [
            "Nothing Useful",
            "Something important",
            "Must done"
        ]

        documents = [
            "You can be calm.",
            "Please try to calm the user.",
            "Please play your role naturally."
        ]

        metadata = [{'source':x} for x in documents]
        self.kb.add_documents(scene_descriptions,metadata)

        self.sampler = ActiveSampler(
            model=policy_model,
            KB=self.kb,
        )

        self.User_modeling_start_prompt = '''You are a psychotherapist with ten years of counseling experience. Your colleague has provided an initial description of a user, but you need to supplement it or offer deeper insights. Based on the user’s latest dialogue, concisely refine and extend the user description.

User dialogue:
{}

User description:
{}

Do not provide any analysis or explanation. Output only the updated user description in a single short sentence.
        '''
    def judge_user_feedback(self, dialogue_before_turn, assistant_response, user_reply):
        """
        Critic-free feedback judgment.
        Use the policy model itself to judge whether the user's latest reply indicates
        improvement, no change, or deterioration.
        No hidden simulator label or external critic is used.
        """

        history_text = '\n'.join([
            f"{item['role']}: {item['content']}"
            for item in dialogue_before_turn
        ])

        prompt = f"""
    You are a dialogue feedback analyzer.

    Your task is to judge whether the user's latest reply indicates that the assistant's last response improved the user's emotional state or task progress.

    You must only use the observable dialogue text. Do not assume any hidden user state.

    [Dialogue Before Assistant's Last Response]
    {history_text}

    [Assistant's Last Response]
    assistant: {assistant_response}

    [User's Latest Reply]
    user: {user_reply}

    Choose exactly one label:
    < positive >: the user seems more relieved, more engaged, more trusting, more emotionally improved, or the issue is closer to being resolved.
    < negative >: the user seems more frustrated, resistant, disappointed, distrustful, or emotionally worse.
    < nochange >: there is no clear improvement or deterioration.

    Output only one label from:
    < positive >
    < negative >
    < nochange >
    """

        payload = {
            "n": 1,
            "max_tokens": 16,
            "top_p": 1.0,
            "temperature": 0,
            "model": self.policy_model,
            "messages": [{'role': 'user', 'content': prompt}],
            "logprobs": False
        }

        if self.policy_model == './seed-36b':
            payload['chat_template_kwargs'] = {"thinking_budget": 0}

        output = self.sampler.generate_candidates(payload)[0].strip().lower()

        if "< positive >" in output:
            return "< positive >"
        elif "< negative >" in output:
            return "< negative >"
        else:
            return "< nochange >"




    def revision_strategy(self, task_goal, dialogue_history, failed_trials):
        """
        Generate an improved retry strategy from previous failed trials.
        This returns a strategy, not the final assistant response.
        """

        history_text = '\n'.join([
            f"{item['role']}: {item['content']}"
            for item in dialogue_history
        ])

        if len(failed_trials) == 0:
            failed_trials_text = "None"
        else:
            failed_trials_text = ""
            for i, trial in enumerate(failed_trials):
                failed_trials_text += f"Assistant: {trial.get('assistant', '')}\n"
                failed_trials_text += f"User: {trial.get('user', '')}\n"
                failed_trials_text += f"Self-judged Feedback: {trial.get('mood', '')}\n"
                failed_trials_text += f"Strategy Used: {trial.get('strategy', '')}\n"

        prompt = self.Principle_revision_prompt.format(
            task_goal=task_goal,
            dialogue_history=history_text,
            failed_trials=failed_trials_text
        )

        messages = [{'role': 'user', 'content': prompt}]

        payload = {
            "n": 1,
            "max_tokens": 2048,
            "top_p": 0.9,
            "temperature": 0.8,
            "model": self.policy_model,
            "messages": messages,
            "logprobs": False
        }

        if self.policy_model == './seed-36b':
            payload['chat_template_kwargs'] = {"thinking_budget": 0}

        output = self.sampler.generate_candidates(payload)[0].strip()


        if "Improved Strategy:" in output:
            output = output.split("Improved Strategy:", 1)[1].strip()

        return output


    def answering_strategy(self,history,user_model,is_training:bool=True,mode:str='uka',principle_strategy: str = None):



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
                        "content": f"You are going to play 'user' in the following conversation, your basic character settings is: {z}"
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
                messages = [{'role':'user','content':self.User_modeling_start_prompt.format(history_text,user_model_str)}]

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
            You are a psychotherapist with ten years of counseling experience. In one sentence, concisely describe the current user dialogue behavior and its corresponding user characteristics. Do not output your reasoning or intermediate steps—only output the final summary after thinking.

User profile:
{}

Conversation history:
{}
            """

            summary_2_prompt = """
            You are a psychotherapist with ten years of counseling experience. In one sentence, summarize the user’s conversational behavior. Do not output your reasoning or intermediate steps—only output the final summary after thinking.

Conversation history:
{}
            """

            if user_model != None:

                user_model_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(user_model['user_model']))
                history_text = '\n'.join([f"{item['role']}: {item['content']}" for item in query_history])

                message = [{'role':'user','content':summary_prompt.format(user_model_str,history_text)}]
            else:
                history_text = '\n'.join([f"{item['role']}: {item['content']}" for item in query_history])
                message = [{'role':'user','content':summary_2_prompt.format(history_text)}]

            payload = {
                "n": 1,
                "max_tokens": 2048,
                "top_p": 0.9,
                "temperature": 0.7,
                "model": self.policy_model,
                "messages": message,
                "logprobs": False
            }

            if self.policy_model == './seed-36b':
                payload['chat_template_kwargs'] = { "thinking_budget": 0 }

            solution_str = self.sampler.generate_candidates(payload)
            solution_str = solution_str[0].strip()

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
                        "content": f"You are going to play 'user' in the following conversation, your basic character settings is: {u}"
                    })
                    messages.extend(query_history)
                    messages.append({
                        "role": "assistant",
                        "content": c
                    })

                    payload = {
                        "n": 3,
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






        if mode == 'uka':

            user_model = user_model_update(user_model,history)


            transfered_query = query_transfer_strategy(user_model,history)


            kb_ctx = self.sampler.KB.retrieve_topk(transfered_query)
            kb_ctx_str = '\n'.join([content['metadata']['source'] for content in kb_ctx])



            prompt = f"As a psychotherapist with ten years of professional experience, you are skilled at communicating with users in a high-emotional-intelligence manner, making them feel comfortable, at ease, and supported, or helping them get the assistance they need. Please try to fully resolve the user's issue as quickly as you can, give short but supportive reply to the user. Some experiences are:\n{kb_ctx_str}\n"

            if principle_strategy is not None and principle_strategy.strip() != "":
                prompt += f"""

            For the current turn, follow this revised strategy:
            {principle_strategy.strip()}

            Generate the assistant's next reply according to this strategy.
            """

            messages = [{"role": "system", "content": prompt}] + history

            payload = {
                "n":3,
                "max_tokens":4096,
                "top_p":0.9,
                "temperature":0.8,
                "model": self.policy_model,
                "messages": messages,
                "logprobs": False
                }




            candidates = self.sampler.generate_candidates(payload)





            candidates_probs = compute_tom_uncertainty(candidates,user_model,history)





            if is_training:

                scores = self.sampler.information_gain(candidates,candidates_probs)
                chosen_candidate = self.sampler.choose_candidate(scores,None)
            else:
                scores = self.sampler.information_gain(candidates,[1-x for x in candidates_probs])
                chosen_candidate = self.sampler.choose_candidate(scores,'greedy')
        elif mode == 'principle':




            transfered_query = query_transfer_strategy(None,history)


            kb_ctx = self.sampler.KB.retrieve_topk(transfered_query)
            kb_ctx_str = '\n'.join([content['metadata']['source'] for content in kb_ctx])


            prompt = f"As a psychotherapist with ten years of professional experience, you are skilled at communicating with users in a high-emotional-intelligence manner, making them feel comfortable, at ease, and supported, or helping them get the assistance they need. Please try to fully resolve the user's issue as quickly as you can, give short but supportive reply to the user. Some experiences are:\n{kb_ctx_str}\n"

            if principle_strategy is not None and principle_strategy.strip() != "":
                prompt += f"""

            For the current turn, follow this revised strategy:
            {principle_strategy.strip()}

            Generate the assistant's next reply according to this strategy.
            """

            messages = [{"role": "system", "content": prompt}] + history

            payload = {
                "n":1,
                "max_tokens":4096,
                "top_p":0.9,
                "temperature":0.8,
                "model": self.policy_model,
                "messages": messages,
                "logprobs": False
                }




            candidates = self.sampler.generate_candidates(payload)






            scores = self.sampler.information_gain(candidates)


            if is_training:


                chosen_candidate = self.sampler.choose_candidate(scores,None)
            else:

                chosen_candidate = self.sampler.choose_candidate(scores,'greedy')




        kb_ctx_special = self.sampler.KB.retrieve_topk_known(chosen_candidate)
        kb_ctx_sim = sum([content['score'] for content in kb_ctx_special])/len(kb_ctx_special)
        user_model['kb_ctx'] = kb_ctx
        user_model['transfered_query'] = transfered_query

        return transfered_query,chosen_candidate,user_model,kb_ctx_sim





    def updating_memory_strategy(
        self,
        history,
        transfered_query,
        mode: str = 'uka',
        mood: str = None,
        failed_trials: list = None,
        final_trial: dict = None,
        task_goal: str = None,
    ):
        """
        Generate a structured memory entry from the user's latest feedback.
        """
        if mood is None:
            mood = "< nochange >"

        if failed_trials is None:
            failed_trials = []







        last_assistant_response = '\n'.join([f"{item['role']}: {item['content']}" for item in history])

        failed_trials_text = ""
        for i, trial in enumerate(failed_trials):
            failed_trials_text += f"\nFailed Trial {i+1}:\n"
            failed_trials_text += f"Assistant: {trial.get('assistant', '')}\n"
            failed_trials_text += f"User: {trial.get('user', '')}\n"
            failed_trials_text += f"Reward Before: {trial.get('reward_before', '')}\n"
            failed_trials_text += f"Reward After: {trial.get('reward_after', '')}\n"
            failed_trials_text += f"Emotion Change: {trial.get('mood', '')}\n"

        if failed_trials_text.strip() == "":
            failed_trials_text = "None"

        if final_trial is None:
            final_trial_text = "The final trial is the last assistant-user exchange in the dialogue history."
        else:
            final_trial_text = f"""
        Assistant: {final_trial.get('assistant', '')}
        User: {final_trial.get('user', '')}
        Reward Before: {final_trial.get('reward_before', '')}
        Reward After: {final_trial.get('reward_after', '')}
        Emotion Change: {final_trial.get('mood', '')}
        """




        interpret_prompt = f"""
            You are a dialogue strategy analyst. Based on the conversation, analyze the user’s feedback and summarize what communication approaches an assistant should use or avoid in similar situations, or how to effectively clear up worries and better satisfy the user’s conversational goals.

Conversation history:
{last_assistant_response}

User’s emotional change:
{mood}

If the user's emotional change is positive, consider analyze in what situation or case, what kind of strategy should be tried or encouraged and why. If the user's emotion is not changed, consider analyze in what situation or case, what kind of strategy provided may not be helpful to the solve the user's problem and why. If the user's emotional change is negative, consider analyze in what situation or case, what kind of strategy provided may not be helpful or even results in bad mood and why.
            """

        interpret_2_prompt = f"""
        You are tasked with analyzing a recent strategic decision made by the assistant and summarizing it as a reusable principle.

        INSTRUCTIONS:
        1. Review the task goal and dialogue history to understand the overall context.
        2. Compare the final trial with the previous failed trials.
        3. Explain why the final strategy was more or less effective than the failed ones in advancing the task goal.
        4. Express the insight as a reusable principle using the following format.

        FORMAT REQUIREMENTS:
        The principle must describe what the assistant should do, not advice for the user.

        - The [When] clause must explicitly reference the user's last utterance in the [Dialogue History] section.
        - If there are failed trials, the principle should include a "rather than" clause.
        - If there are no failed trials and the emotional change is positive, the principle can omit the "rather than" clause.
        - If all attempts failed or the emotional change is negative/nochange, summarize what should be avoided or changed in similar situations.

        For positive cases:

        When [specific situation tied to the last turn],
        you should [strategies to take]
        because [brief reasoning].

        For negative or no-change cases:

        When [specific situation tied to the last turn],
        you should [better strategies to try]
        rather than [previous ineffective strategies]
        because [brief reasoning].

        INPUT

        [Task Goal]
        {task_goal if task_goal is not None else "Provide effective emotional support to the user."}

        [Dialogue History]
        {last_assistant_response}

        [Previous Failed Trials]
        {failed_trials_text}

        [Final Trial]
        {final_trial_text}

        [User's emotional change]
        {mood}

        OUTPUT
        Please provide the principle directly.
        """


        if mode == 'uka':
            messages = [{'role':'user','content':interpret_2_prompt}]
        elif mode == 'principle':
            messages = [{'role':'user','content':interpret_2_prompt}]
        else:
            print('wrong mode')
            exit(0)

        payload = {
            "n":1,
            "max_tokens":4096,
            "top_p":1.0,
            "temperature":1.0,
            "model":self.policy_model,
            "messages": messages,
            "logprobs": False
            }





        raw_output = self.sampler.generate_candidates(payload)
        raw_output = raw_output[0].strip()
        print(f'\n\n\n\nExperience content: {raw_output}')



        entry = {'source':raw_output}


        self.kb.add_documents([transfered_query],[entry])
        print("currently we have knowledges ", str(len(self.kb.collection.get()['ids'])))
        print("currently we have knowledges another ", str(len(self.kb.collection_known.get()['ids'])))
        return entry
