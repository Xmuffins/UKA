import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from active_response import Active_Responser
from simulator_response import chat_player, player_init


MAX_TURNS = 9
MAX_WORKERS = 4


def split_assistant_response(query_result):
    if "</seed:think>" in query_result:
        think, content = query_result.split("</seed:think>", 1)
        return content, think
    if "</think>" in query_result:
        think, content = query_result.split("</think>", 1)
        return content, think
    return query_result, ""


def semantic_eval(policy_model, knowledge_base_state):
    current_dir = os.path.dirname(__file__)
    profile_path = os.path.join(
        current_dir,
        "profile",
        "simulator_profile_withfirsttalk.jsonl",
    )

    with open(profile_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print("policy_model:", policy_model)
    print("knowledge_base_state:", knowledge_base_state)

    store_file = os.path.join(current_dir, f"{policy_model}-uka-normal.jsonl")
    if not os.path.exists(store_file):
        open(store_file, "w", encoding="utf-8").close()

    responser = Active_Responser(policy_model)

    with open(store_file, "r", encoding="utf-8") as f:
        finished_ids = {json.loads(line)["id"] for line in f if line.strip()}

    def run_session(simulator):
        if simulator["id"] in finished_ids:
            return None

        player = player_init(id=simulator["id"])

        print(player["player"])
        print(player["scene"])
        print(player["character"])
        print(player["task"])
        print("player:{}".format(simulator["first_talk"]))

        player["user_model"] = {
            "user_model": [],
            "scores": [],
            "probs": [],
            "entropy": [],
        }
        player["self_model"] = ""
        transfered_query = ""

        for turn in range(1, MAX_TURNS + 1):
            if turn == 1:
                player["history"].append(
                    {
                        "role": "user",
                        "content": simulator["first_talk"],
                        "emotion-point": player["emo_point"],
                    }
                )
            else:
                player = chat_player(player)
                latest_user = player["history"][-1]
                print("player:{}".format(latest_user["content"]))
                print("player-emotion:{},{}".format(player["emo_point"], player["emo_state"]))
                print(latest_user)
                print()

                if "再见" in latest_user["content"] or "拜拜" in latest_user["content"]:
                    break
                if latest_user["emotion-point"] >= 100 or latest_user["emotion-point"] < 10:
                    break

                if transfered_query and knowledge_base_state == "training":
                    responser.updating_memory_strategy(player["history"], transfered_query)

            if knowledge_base_state == "training":
                transfered_query, query_result, player["user_model"] = responser.answering_strategy(
                    player["history"],
                    player["user_model"],
                )
            else:
                transfered_query, query_result, player["user_model"] = responser.answering_strategy(
                    player["history"],
                    player["user_model"],
                    False,
                )

            print("npc response:", query_result)
            print("temp:", transfered_query)
            print("user_model:", player["user_model"])

            content, think = split_assistant_response(query_result)
            player["history"].append(
                {
                    "role": "assistant",
                    "content": content,
                    "think": think,
                }
            )

        return player

    total = len(data)
    done = 0
    train_size = total // 5
    simulators = data[:train_size] if knowledge_base_state == "training" else data[train_size:]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_session, simulator) for simulator in simulators]
        for future in as_completed(futures):
            try:
                player = future.result()
            except Exception as exc:
                print(f"Task failed: {exc}")
                continue

            done += 1
            print(f"Progress: {done}/{len(simulators)} ({done / max(len(simulators), 1):.2%})")
            if player is None:
                continue

            if knowledge_base_state != "training":
                with open(store_file, "a", encoding="utf-8") as file:
                    file.write(json.dumps(player, ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_sentient_eval.py <policy_model> <training|testing>")
        sys.exit(1)

    policy_model = sys.argv[1]
    knowledge_base_state = sys.argv[2]
    semantic_eval(policy_model, knowledge_base_state)


if __name__ == "__main__":
    main()
