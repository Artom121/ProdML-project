from pathlib import Path

import pandas as pd
import yaml

from features import build_features, load_dialogs

ROOT = Path(__file__).parent
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
raw_dir = ROOT / cfg["raw_dir"]
out_dir = (ROOT / cfg["processed_dir"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "test"]:
    messages = load_dialogs(raw_dir / f"{split}.json")
    participants = pd.read_csv(raw_dir / f"y{split}.csv")
    table = build_features(messages, participants)
    table.to_csv(out_dir / f"{split}.csv", index=False)
    print(f"{split}: {len(table)} участников, {table.shape[1]} колонок -> {out_dir / split}.csv")
