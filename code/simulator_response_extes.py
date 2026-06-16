import random
import time
import json
import copy
import os

random.seed(123)
from openai import OpenAI


def call_llm(prompt):
    api_key = os.getenv("EXTES_SIMULATOR_API_KEY") or os.getenv("SIMULATOR_API_KEY")
    if not api_key:
        raise RuntimeError("Set EXTES_SIMULATOR_API_KEY or SIMULATOR_API_KEY before running the simulator.")

    base_url = os.getenv(
        "EXTES_SIMULATOR_BASE_URL",
        os.getenv("SIMULATOR_BASE_URL", "https://www.autodl.art/api/v1/"),
    )
    model = os.getenv("EXTES_SIMULATOR_MODEL", os.getenv("SIMULATOR_MODEL", "gpt-5.4-mini"))

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={
            "include_usage": True
        },
    )

    reasoning_content = ""
    answer_content = ""
    is_answering = False

    for chunk in completion:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta


        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content


        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_content += delta.content

    return answer_content

def player_init(id = None,data={}):

    player_data = {
        "id":id,
        "emo_point": 0,

        "player": data[id]["scene"],
        "scene": data[id]["scene"],
        'emotion_type': data[id]["emotion_type"],



        "history": []
    }

    return player_data


def planning_reply(player_data):
    template = """Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation. You can only reply with one of the following sentences:
    A. No, the Patient feels worse.
    B. No, the Patient feels the same.
    C. No, but the Patient feels better.
    D. Yes, the Patient’s issue has been solved.

    If you believe that the patient’s problem has been resolved, please choose D. If you believe that the patient’s problem has not been resolved, but his emotional issue has been somewhat alleviated compared to the last conversation turn, you can choose C. If you believe that the patient’s emotional state has worsened compared to the last conversation turn, you can choose A. Otherwise, if the patient’s emotional state remains unchanged, please choose B.

    Inputs
    The following is a dialogue about {{emotion_type}}.
    Situation of the Patient: {{scene}}
    Dialogue: {{dialog_history}}

    Output format (strict)
    Decision:
    [Your Decision (Only reply one of '<A>', '<B>', '<C>', '<D>')]
"""





    prompt = template.replace("{{scene}}",player_data['scene']).replace("{{emotion_type}}",player_data['emotion_type'])


    history = player_data["history"]
    history_str = []
    new_his_str = ""
    mapping = {"user": "You", "assistant": "Patient"}
    for mes in history:
        history_str.append({"role": mapping[mes["role"]], "content": mes["content"]})
    history_str = json.dumps(history_str, ensure_ascii=False, indent=2)
    prompt = prompt.replace("{{dialog_history}}",history_str)


    while True:
        try:

            replys = []
            for i in range(10):
                reply = call_llm(prompt)
                time.sleep(0.3)
                replys.append(reply)


            reward = []
            print(replys)
            for rep in replys:





                final_answer = rep.replace('Decision','')
                if "A" in final_answer:
                    reward.append(-1.0)
                elif "B" in final_answer:
                    reward.append(-0.5)
                elif "C" in final_answer:
                    reward.append(0.5)
                elif "D" in final_answer:
                    reward.append(1.0)


            if len(reward) == 0:
                player_data['emo_point'] = 0
            else:
                player_data['emo_point'] = sum(reward)/len(reward)





            if reply is not None:
                break
        except Exception as e:
            print(e)
            time.sleep(3)












    return player_data

def player_reply(player_data):

    template = """Now enter the role-playing mode. In the following conversation, you will play as a patient in a counselling conversation with a therapist.

[Task]
Using:
- scene setup
- dialogue history
- the Patient’s latest reply
generate a natural, realistic User reply.

[Inputs]
Scene: {{scene}}
History: {{dialog_history}}
Latest turns: {{new_history}}

[Output format (strict)]
Response:
[Final reply (only 1 short message)]
"""



    emo_point = player_data['emo_point']
    history = player_data["history"]


    prompt = template.replace("{{scene}}",player_data['scene']).replace("{{player_topic}}",player_data["scene"])


    if not history:
        prompt = prompt.replace("{{dialog_history}}","The conversation starts. You are the user/player. Begin by initiating a topic and open up with a short message.").replace("{{new_history}}","")
    else:
        history_str = []
        new_his_str = []
        mapping ={"user":"You","assistant":"Patient"}

        for mes in history[:-2]:
            history_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
        history_str=json.dumps(history_str, ensure_ascii=False, indent=2)

        for mes in history[-2:]:
            new_his_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
        new_his_str=json.dumps(new_his_str, ensure_ascii=False, indent=2)

        prompt = prompt.replace("{{dialog_history}}",history_str).replace("{{new_history}}",new_his_str)

    reply = None

    while True:
        try:

            reply = call_llm(prompt)


            reply = reply.split("Response:")[-1].strip("\n").strip("[").strip("]").strip("“").strip("”")
            if reply is not None:
                break
        except Exception as e:
            print(e)
            time.sleep(3)


    history = history + [{"role": "user", "content": reply,"emotion-point":emo_point}]
    player_data['history'] = history

    return player_data


def chat_player(player_data):
    temp_data = copy.deepcopy(player_data)


    if temp_data['history']!=[]:
        temp_data = planning_reply(temp_data)
    else:
        planning = {}

    temp_data = player_reply(temp_data)

    return temp_data

