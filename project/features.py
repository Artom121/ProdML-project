import json
import re

import numpy as np
import pandas as pd

CYR = re.compile(r"[а-яё]", re.IGNORECASE)
LAT = re.compile(r"[a-z]", re.IGNORECASE)
PUNCT = re.compile(r"[^\w\s]")
LETTER = re.compile(r"[^\W\d_]")
MARKDOWN = re.compile(r"(\*\*|__|```|^#{1,6}\s|^\s*[-*•]\s|^\s*\d+[.)]\s)", re.MULTILINE)
BOT_WORDS = re.compile(r"бот|\bbot|нейрос|нейронк|\bии\b|\bai\b|gpt|llm|робот|robot", re.IGNORECASE)


def load_dialogs(path):
    # json -> таблица, одна строка на сообщение
    with open(path, encoding="utf-8") as f:
        dialogs = json.load(f)
    rows = [(d, m["message"], m["participant_index"], m["text"]) for d, msgs in dialogs.items() for m in msgs]
    df = pd.DataFrame(rows, columns=["dialog_id", "message", "participant_index", "text"])
    return df.sort_values(["dialog_id", "message"])


def norm(text):
    return " ".join(text.lower().split())


def jaccard(a, b):
    # похожесть двух реплик по символьным триграммам
    a, b = f" {norm(a)} ", f" {norm(b)} "
    x = {a[i : i + 3] for i in range(len(a) - 2)}
    y = {b[i : i + 3] for i in range(len(b) - 2)}
    return len(x & y) / len(x | y) if x | y else 0.0


def mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def participant_features(turns, p):
    # turns - список (participant_index, text) по порядку, p - участник, для которого считаем
    texts = [t for who, t in turns if who == p]
    lens = [len(t) for t in texts]
    joined = "\n".join(texts)
    n_cyr, n_lat = len(CYR.findall(joined)), len(LAT.findall(joined))
    with_letters = [t for t in texts if LETTER.search(t)]

    # как участник реагирует на прошлую реплику собеседника
    echo, sim, len_ratio, double = [], [], [], []
    for (prev_who, prev), (who, text) in zip(turns, turns[1:]):
        if who != p:
            continue
        double.append(prev_who == p)
        if prev_who != p:
            echo.append(norm(text) == norm(prev))
            sim.append(jaccard(text, prev))
            len_ratio.append(np.log((len(text) + 1) / (len(prev) + 1)))

    return {
        "n_msgs": len(texts),
        "share_msgs": len(texts) / len(turns),
        "is_first": float(turns[0][0] == p),
        "is_last": float(turns[-1][0] == p),
        "len_mean": mean(lens),
        "len_max": max(lens, default=np.nan),
        "len_min": min(lens, default=np.nan),
        "len_total": sum(lens),
        "frac_question": mean(["?" in t for t in texts]),
        "frac_end_punct": mean([t.rstrip()[-1:] in (".", "!", "?") for t in texts]),
        "frac_start_upper": mean([LETTER.search(t).group().isupper() for t in with_letters]),
        "frac_all_lower": mean([t == t.lower() for t in with_letters]),
        "frac_markdown": mean([bool(MARKDOWN.search(t)) for t in texts]),
        "frac_bot_words": mean([bool(BOT_WORDS.search(t)) for t in texts]),
        "cyr_ratio": n_cyr / (n_cyr + n_lat) if n_cyr + n_lat else np.nan,
        "punct_ratio": len(PUNCT.findall(joined)) / max(len(joined), 1),
        "echo_rate": mean(echo),
        "sim_prev_mean": mean(sim),
        "sim_prev_max": max(sim, default=np.nan),
        "len_ratio_prev": mean(len_ratio),
        "double_msg_rate": mean(double),
    }


def build_features(messages, participants):
    # одна строка на участника: его текст, признаки и такие же признаки собеседника (opp_*)
    turns = {d: list(zip(g["participant_index"], g["text"])) for d, g in messages.groupby("dialog_id")}
    rows = []
    for d, p in zip(participants["dialog_id"], participants["participant_index"]):
        row = participant_features(turns[d], p)
        row["text"] = "\n".join(t for who, t in turns[d] if who == p)
        rows.append(row)
    feats = pd.DataFrame(rows)
    table = pd.concat([participants.reset_index(drop=True), feats], axis=1)

    own = [c for c in feats.columns if c != "text"]
    opp = table[["dialog_id", "participant_index"] + own].copy()
    opp["participant_index"] = 1 - opp["participant_index"]
    opp = opp.rename(columns={c: "opp_" + c for c in own})
    return table.merge(opp, on=["dialog_id", "participant_index"], how="left")
