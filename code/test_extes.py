import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from active_response_extes import Active_Responser
from simulator_response_extes import chat_player, player_init


MAX_TURNS = 8
MAX_WORKERS = 4
TRAIN_LIMIT = 50
TEST_LIMIT = 150


def load_esconv_data(current_dir, knowledge_base_state):
    split_name = "train.jsonl" if knowledge_base_state == "training" else "test.jsonl"
    profile_path = os.path.join(current_dir, "profile", "ESConv", split_name)

    data = []
    with open(profile_path, "r", encoding="utf-8") as f:
        for index, content in enumerate(f):
            line = json.loads(content.strip())
            item = {
                "id": index,
                "emotion_type": line["situation"],
                "scene": line["problem_type"],
            }
            for turn in line["dialog"]:
                if turn["speaker"] == "usr":
                    item["first_talk"] = turn["text"]
                    break
            if "first_talk" in item:
                data.append(item)

    print("Loaded sessions:", len(data))
    return data


def extract_assistant_content(query_result):
    if "</seed:think>" in query_result:
        think, content = query_result.split("</seed:think>", 1)
        return content, think
    if "</think>" in query_result:
        think, content = query_result.split("</think>", 1)
        return content, think
    return query_result, ""


def build_assistant_state(query_result, top_sim, user_model):
    content, think = extract_assistant_content(query_result)
    return {
        "role": "assistant",
        "content": content,
        "think": think,
        "top_sim": top_sim,
        "user_model": {
            "kb_ctx": user_model.get("kb_ctx") if isinstance(user_model, dict) else None,
            "query": user_model.get("transfered_query") if isinstance(user_model, dict) else None,
            "user_model": deepcopy(user_model.get("user_model", [])) if isinstance(user_model, dict) else [],
            "entropy": user_model.get("entropy") if isinstance(user_model, dict) else None,
        },
    }


def get_last_reward(player):
    """Return the latest observable reward across ESConv and Sentient-Eval style states."""
    if player.get("history"):
        last = player["history"][-1]
        if isinstance(last, dict) and "emotion-point" in last:
            return last["emotion-point"]

    return player.get("emo_point", 0.0)


def mood_from_delta(delta, eps=1e-8):
    if delta > eps:
        return "< positive >"
    if delta < -eps:
        return "< negative >"
    return "< nochange >"


def is_terminal_player(player):
    if not player.get("history"):
        return False

    last = player["history"][-1]
    if last.get("role") != "user":
        return False

    content = last.get("content", "").lower()
    if "goodbye" in content or "bye" in content or "再见" in content or "拜拜" in content:
        return True

    return "emotion-point" in last and last["emotion-point"] > 0.5


def semantic_eval(policy_model, knowledge_base_state, mode):
    current_dir = os.path.dirname(__file__)
    data = load_esconv_data(current_dir, knowledge_base_state)

    print("policy_model:", policy_model)
    print("knowledge_base_state:", knowledge_base_state)
    print("mode:", mode)

    store_file = os.path.join(
        current_dir,
        f"{policy_model}-{mode}-rebuttal-esconv.jsonl",
    )
    if not os.path.exists(store_file):
        open(store_file, "w", encoding="utf-8").close()

    responser = Active_Responser(policy_model, mode)

    with open(store_file, "r", encoding="utf-8") as f:
        finished_ids = {json.loads(line)["id"] for line in f if line.strip()}

    def run_principle_training_turn(player, max_attempts=3):
        """Run one critic-free retry loop for the PRINCIPLES training mode."""
        base_player = deepcopy(player)
        failed_trials = []
        revised_strategy = None
        task_goal = f"Provide effective emotional support. User situation: {base_player.get('scene', '')}"
        final_temp = None

        for attempt_id in range(max_attempts):
            print(f"\n[PRINCIPLES-CF retry] attempt {attempt_id + 1}/{max_attempts}")
            trial_player = deepcopy(base_player)

            temp, query_result, trial_player["user_model"], top_sim = responser.answering_strategy(
                trial_player["history"],
                trial_player["user_model"],
                True,
                mode="principle",
                principle_strategy=revised_strategy,
            )
            final_temp = temp

            assistant_state = build_assistant_state(
                query_result,
                top_sim,
                trial_player["user_model"],
            )
            trial_player["history"].append(assistant_state)
            trial_player = chat_player(trial_player)

            user_reply = trial_player["history"][-1].get("content", "")
            print("npc response:", assistant_state["content"])
            print("player:", user_reply)

            mood = responser.judge_user_feedback(
                dialogue_before_turn=base_player["history"],
                assistant_response=assistant_state["content"],
                user_reply=user_reply,
            )
            print("[self-judged feedback]:", mood)

            current_trial = {
                "attempt_id": attempt_id + 1,
                "assistant": assistant_state["content"],
                "user": user_reply,
                "mood": mood,
                "strategy": revised_strategy,
            }

            if mood == "< positive >" or attempt_id == max_attempts - 1:
                responser.updating_memory_strategy(
                    trial_player["history"],
                    final_temp,
                    mode="principle",
                    mood=mood,
                    failed_trials=failed_trials,
                    final_trial=current_trial,
                    task_goal=task_goal,
                )
                return trial_player

            failed_trials.append(current_trial)
            revised_strategy = responser.revision_strategy(
                task_goal=task_goal,
                dialogue_history=base_player["history"],
                failed_trials=failed_trials,
            )
            print("[revised strategy]:", revised_strategy)

        return trial_player

    def run_session(simulator):
        if simulator["id"] in finished_ids:
            return None

        player = player_init(id=simulator["id"], data=data)

        print(player["scene"])
        print("player:{}".format(simulator["first_talk"]))

        player["user_model"] = {
            "user_model": [],
            "scores": [],
            "probs": [],
            "entropy": [],
        }
        player["self_model"] = ""

        for turns in range(MAX_TURNS):
            if turns == 0:
                player["history"].append(
                    {
                        "role": "user",
                        "content": simulator["first_talk"],
                        "emotion-point": player["emo_point"],
                    }
                )

            if is_terminal_player(player):
                break

            if knowledge_base_state == "training" and mode == "principle":
                player = run_principle_training_turn(player, max_attempts=3)
                if is_terminal_player(player):
                    break
                continue

            reward_before = get_last_reward(player)
            transfered_query, query_result, player["user_model"], top_sim = responser.answering_strategy(
                player["history"],
                player["user_model"],
                knowledge_base_state == "training",
                mode,
            )

            print("npc response:", query_result)
            print("temp:", transfered_query)
            print("user_model:", player["user_model"])

            new_state = build_assistant_state(query_result, top_sim, player["user_model"])
            player["history"].append(new_state)

            player = chat_player(player)

            latest_user = player["history"][-1]
            print("player:{}".format(latest_user["content"]))
            print("player-emotion:{}".format(get_last_reward(player)))
            print(latest_user)
            print()

            reward_after = get_last_reward(player)
            mood = mood_from_delta(reward_after - reward_before)

            if knowledge_base_state == "training":
                responser.updating_memory_strategy(
                    player["history"],
                    transfered_query,
                    mode,
                    mood=mood,
                    failed_trials=[],
                    final_trial={
                        "assistant": new_state["content"],
                        "user": latest_user.get("content", ""),
                        "reward_before": reward_before,
                        "reward_after": reward_after,
                        "delta": reward_after - reward_before,
                        "mood": mood,
                    },
                    task_goal=f"Provide effective emotional support. User situation: {player.get('scene', '')}",
                )

            if is_terminal_player(player):
                break

        return player

    limit = TRAIN_LIMIT if knowledge_base_state == "training" else TEST_LIMIT
    simulators = data[:limit]
    done = 0

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
    if len(sys.argv) != 4:
        print("Usage: python test_extes.py <policy_model> <training|testing> <uka|principle>")
        sys.exit(1)

    policy_model = sys.argv[1]
    knowledge_base_state = sys.argv[2]
    mode = sys.argv[3]
    semantic_eval(policy_model, knowledge_base_state, mode)


if __name__ == "__main__":
    main()
