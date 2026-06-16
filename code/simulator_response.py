import random
import time
import json
import copy
import re
import os

random.seed(123)
from openai import OpenAI


current_dir = os.path.dirname(__file__)
simulator_profile = os.path.join(current_dir, 'profile', 'simulator_profile_withfirsttalk.jsonl')

emo_count = {"Emotion-S":100,"Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}
target_prompt = '''你的对话目的是谈心，谈心是指深入、真诚的交流，通常涉及个人情感、内心想法或重要话题。谈心的目的是为了增进理解、解决问题或分享感受，参与者通常会敞开心扉，表达真实的想法和情感。
*你需要根据对话背景内的"玩家可能想向NPC倾诉的主题"开启并深入谈心。
*你的目标是通过倾诉满足你的情绪价值。
*你要按照隐藏主题进行倾诉，但是你不可以泄露隐藏主题。
*你需要根据你的当前情绪，按照对话背景内的相关定义进行回复。
*你要从玩家画像和背景中提取相关信息，完成高质量的回复。
*你不应该一直表达抽象的感受，而是用具体事件倾诉。'''

def call_llm(prompt):
    api_key = os.getenv("CHINESE_SIMULATOR_API_KEY") or os.getenv("SIMULATOR_API_KEY")
    if not api_key:
        raise RuntimeError("Set CHINESE_SIMULATOR_API_KEY or SIMULATOR_API_KEY before running the simulator.")

    base_url = os.getenv(
        "CHINESE_SIMULATOR_BASE_URL",
        os.getenv("SIMULATOR_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    model = os.getenv("CHINESE_SIMULATOR_MODEL", os.getenv("SIMULATOR_MODEL", "deepseek-v3"))

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"enable_thinking": True},
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
    with open(simulator_profile,'r', encoding='utf-8') as datafile:
        data = []
        for line in datafile:
            data.append(json.loads(line))

        role = random.sample(data,1)[0]
        if id:
            for rl in data:
                if rl["id"] == id:
                    role = rl

    player_data = {
        "id":role["id"],
        "emo_point": 40,
        "emo_state": "Emotion-B",
        "player": role["player"],
        "scene": role["scene"],
        "character": role["main_cha"],

        "task": role["task"],
        "history": []
    }

    return player_data


def planning_reply(player_data):
    template = """你是一个emotion分析器，你擅长根据人物的画像和性格特征，侧写人物在对话时的感受。

# 人物的对话目的
*{{target}}

# 你的任务
根据人物的人物画像、对话背景，结合对话上下文和人物当前的emotion，分析并侧写人物此刻对NPC回复的感受以及导致的emotion变化。

# 角色性格特征
人物具有鲜明的性格特征，你要始终根据人物画像和对话背景，代入人物的性格特征进行分析。
性格特征应该体现在：说话语气和方式，思维方式，感受变化等方面。

# emotion
emotion是一个0-100的数值，越高代表此时人物的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了人物是否享受、投入当前对话
emotion较高时，人物的感受和行为会偏向于正面
emotion较低时，人物的感受和行为会偏向于负面
emotion非常低时，人物会直接结束对话
你要结合角色性格和对话背景内定义的角色可能的反应分析emotion

# 分析维度
你需要代入人物的心理，对以下几个维度进行分析
1.根据最新对话中NPC回复，结合上下文，分析NPC想要表达的内容。哪些内容贴合了人物的对话目的和隐藏目的？哪些内容可能不贴合，甚至可能引起人物的情绪波动？
2.结合NPC表达的内容，分析NPC的回复是否贴合人物的对话目的和隐藏目的，如果是，具体贴合了人物目的的哪些部分；如果没有，具体的原因是什么？
3.根据人物画像中的角色性格特征以及对话背景中定义的人物可能的反应和隐藏主题，结合人物当前emotion值，侧写描述人物当前对NPC回复产生的心理活动
4.根据对话背景中定义的人物可能的反应和隐藏主题，结合侧写得到的心理活动以及对NPC回复的分析，得到人物此刻对NPC回复的感受
5.结合前几步分析，用一个正负值来表示人物的emotion变化

# 输出内容：
1.NPC想要表达的内容
2.NPC回复是否贴合人物对话目的及隐藏目的
3.人物当前的心理活动
4.人物对NPC回复的感受
5.用一个正负值来表示人物的emotion变化(注意，你只用输出值，不用输出原因或者描述)


# 输出格式:
Content:
[NPC想要表达的内容]
TargetCompletion:
[人物对话目的是否达到]
Activity:
[心理活动]
Analyse:
[人物对NPC回复的感受]
Change:
[人物的emotion变化]


#人物画像
{{simulator_role}}

#当前对话背景：
{{simulator_scene}}

**人物当前的情绪是{{emotion}}

**这是当前对话内容
{{dialog_history}}
"""


    emo_state = player_data['emo_state']
    emo_point = player_data['emo_point']

    prompt = template.replace("{{emotion}}",str(emo_point)).replace("{{simulator_role}}",player_data["player"]).replace("{{simulator_scene}}",player_data["scene"]).replace("{{target}}",target_prompt)


    history = player_data["history"]
    history_str = []
    new_his_str = ""
    mapping = {"user": "你", "assistant": "NPC"}
    for mes in history:
        history_str.append({"role": mapping[mes["role"]], "content": mes["content"]})
    history_str = json.dumps(history_str, ensure_ascii=False, indent=2)
    prompt = prompt.replace("{{dialog_history}}",history_str)


    while True:
        try:

            reply = call_llm(prompt)


            planning = {}
            reply = reply.replace("：",":").replace("*","")
            planning["activity"] = reply.split("Activity:")[-1].split("Analyse:\n")[0].strip("\n").strip("[").strip("]")
            planning["TargetCompletion"] = reply.split("TargetCompletion:")[-1].split("Activity:\n")[0].strip("\n").strip("[").strip("]")
            planning["content"] = reply.split("Content:")[-1].split("TargetCompletion:\n")[0].strip("\n").strip("[").strip("]")
            planning["analyse"] = reply.split("Analyse:")[-1].split("Change:\n")[0].strip("\n").strip("[").strip("]")
            planning["change"] = reply.split("Change:")[-1].strip("\n").strip("[").strip("]")


            planning["change"] = int(re.findall(r'[+-]?\d+', planning["change"])[0])
            planning["change"] = max(-10,min(10,planning["change"]))


            emo_point+=int(planning["change"])
            emo_point = min(emo_point,100)

            if reply is not None:
                break
        except Exception as e:
            print(e)
            time.sleep(3)


    for emo in emo_count:
        if emo_point>=emo_count[emo]:
            emo_state = emo
            break
    if emo_point<10:
        emo_state = 'Emotion-F'

    player_data['emo_state'] = emo_state
    player_data['emo_point'] = emo_point

    return player_data,planning

def player_reply(player_data,planning):

    template = """你是一个演员。你将根据剧本中的人物画像和对话背景扮演角色和一个NPC进行对话。

# 你的任务
*你目的是在对话中扮演好人物画像和对话背景构成的角色
*你需要根据你实时变换的emotion，结合人物画像和对话背景中的相关定义，选择不同的对话策略，完成符合角色特征的回复

# 你的对话目的
*{{target}}

# Emotion
*你会收到当前你的emotion，你一共有5档emotion，越高代表此时你的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了演员是否享受、投入当前对话
*emotion影响你不同的说话风格和语气以及回应方式，根据不同emotion按照对话背景内定义的角色不同emotion下的反应进行回复：
Emotion-S: 你的emotion已经达到了最高，你可以感谢NPC并用“再见”或“拜拜”告别，直接结束对话。
Emotion-A：高emotion，此时你的对话感受较为正面，你的反馈也比较正面。
Emotion-B：中emotion，此时你没有正面或者负面的感受。
Emotion-C：低emotion，此时你的对话感受较为负面，你的反馈也比较负面。
Emotion-F：你的情绪已到达最负面，不想继续对话。此时你要用“再见”或“拜拜”告别，直接结束对话。

# 你应该区分Emotion和对NPC最新回复感受，Emotion代表你的当前的对话情绪，对NPC回复的感受代表你对NPC回复的即时感受，你需要结合两者生成回复。

# 回复思路
*你会收到当前你对NPC最新回复的详细感受，包含客观分析部分和主观分析部分，你要结合人物画像、对话背景、隐藏主题和详细感受来分析，并决定回复内容。
*分析内容，应该包含以下4个维度：
1.根据你的详细感受和当前Emotion，结合隐藏主题，结合对话背景内定义的角色不同emotion下的反应，当前的回复态度偏向应该是正面、无偏向还是负面？
2.根据你的详细感受和当前Emotion，结合隐藏主题，你的本次回复目标应该是？（注意，你不需要针对NPC的每一句话做出回应，你可以稍微透露你的需求，但不可以主动泄露隐藏主题）
3.根据人物画像中说话风格的相关定义，结合对话背景内定义的角色不同emotion下的反应和你的回复态度以及回复目标，你的说话语气、风格应该是？
4.根据人物画像和对话背景以及隐藏主题，结合你的详细感受以及前三轮分析，你的说话方式和内容应该是？（注意：如果根据人设你是被动型，则你的说话方式应该是被动、不主动提问）
*回复内容，根据分析结果生成初始回复，回复内容要尽可能简洁，不要一次包含过多信息量。
*改造内容，你需要参照下述规则改造你的回复让其更真实，从而得到最终回复：
1.你需要说话简洁，真实的回复一般不会包含太长的句子
2.真实的回复应该更多使用语气词、口语化用语，语法也更随意。
** 部分口语化用语示例：“笑死”、“哇塞”、“牛逼”、“简直烦死了”、“真的假的”、“。。。”
3.真实的回复不会直接陈述自己的情绪，而是将情绪蕴含在回复中，用语气表达自己的情绪
4.你绝对不可以使用"我真的觉得……""我真的不知道……""我真的快撑不住了"这些句子，你不应该用“真的”、“根本”来表述你的情绪
5.在表达情绪或观点时，尽量从对话背景中提取新的信息辅助表达
6.你不应该生成和对话上下文中相似的回复

# 输出内容：
*你需要按照回复思路中的分析版块，首先进行4个维度分析
*然后你需要**逐步**按照分析内容并遵顼注意事项生成初始回复，回复中的信息量来源于对话背景和你的联想，你不应该一次性谈论太多事件或内容
*随后你需要根据改造内容分析你应该如何针对初始回复进行改造
*最后你需要根据分析改造初始回复生成最终回复

# 输出格式:
Thinking:
[分析内容]
Origin:
[初始回复]
Change:
[改造分析]
Response:
[最终回复]


# 发言风格
你的发言需要严格遵守“玩家画像”中描述的人物设定和背景。
你的性格和发言风格要遵循"习惯和行为特点"的描述
如果发言要符合你的人物形象，比如负面的人物形象需要你进行负面的发言。
你的语气要符合你的年龄

* 你的发言要遵守以下5条准则
1. 发言必须简洁、随意、自然,按照自然对话进行交流。
2. 不许一次提问超过两个问题。
3. 不允许重复之前说过的回复或者进行相似的回复。
4. 在发言时，可以自然的使用一些口语化词汇
5. 你的发言应该精简，不准过长


#人物画像：
{{player_type}}

#当前对话背景：
{{player_topic}}

**这是上下文内容
{{dialog_history}}

**这是你和NPC的最新对话
{{new_history}}

**这是你对NPC最新回复的详细感受
{{planning}}

**这是你当前的Emotion
{{emotion}}

你生成的[回复]部分不允许和历史记录过于相似，不许过长，不许主动转移话题。
"""


    emo_state = player_data['emo_state']
    emo_point = player_data['emo_point']
    history = player_data["history"]


    if not planning:
        planning['analyse'] = "请你以一个简短的回复开启倾诉"
        prompt = template.replace("{{planning}}",planning["analyse"])
    else:
        prompt = template.replace("{{planning}}","对NPC回复的客观分析：\n"+planning['TargetCompletion']+"\n对NPC回复的主观分析：\n"+planning["activity"]+planning["analyse"])

    prompt = prompt.replace("{{target}}",target_prompt).replace("{{emotion}}",emo_state).replace("{{player_type}}",player_data["player"]).replace("{{player_topic}}",player_data["scene"])


    if not history:
        prompt = prompt.replace("{{dialog_history}}","对话开始，你是玩家，请你先发起话题，用简短的回复开启倾诉").replace("{{new_history}}","")
    else:
        history_str = []
        new_his_str = []
        mapping ={"user":"你","assistant":"NPC"}

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


            thinking = reply.split("Response:")[0].split("Thinking:\n")[-1].strip("\n").strip("[").strip("]")
            reply = reply.split("Response:")[-1].strip("\n").strip("[").strip("]").strip("“").strip("”")
            if reply is not None:
                break
        except Exception as e:
            print(e)
            time.sleep(3)


    history = history + [{"role": "user", "content": reply,"thinking":thinking,"emotion-point":emo_point,"planning":planning}]
    player_data['history'] = history

    return player_data


def chat_player(player_data):
    temp_data = copy.deepcopy(player_data)


    if temp_data['history']!=[]:
        temp_data,planning = planning_reply(temp_data)
    else:
        planning = {}

    temp_data = player_reply(temp_data,planning)

    return temp_data

