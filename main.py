"""
main.py
-------
Command-line entry point for the Social Media Sentiment Analysis project.

Commands
--------
  python main.py generate    → Generate synthetic dataset
  python main.py train       → Train the ML model
  python main.py predict     → Run interactive CLI predictor
  python main.py evaluate    → Print evaluation metrics
  python main.py dashboard   → Launch Streamlit dashboard
  python main.py all         → Run full pipeline end-to-end
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BANNER = r"""
 ____            _   _                      _   ___ ___
/ ___|  ___ _ __ | |_(_)_ __ ___   ___ _ __ | |_|_ _/ _ \
\___ \ / _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __|| | | | |
 ___) |  __/ | | | |_| | | | | | |  __/ | | | |_ | | |_| |
|____/ \___|_| |_|\__|_|_| |_| |_|\___|_| |_|\__|___\__\_\

  Social Media Sentiment Analysis Dashboard  ·  v2.0
"""


def cmd_generate():
    print("[generate] Creating synthetic dataset …")
    from data.generate_dataset import generate
    import os
    df = generate()
    path = os.path.join(ROOT, "data", "social_media_posts.csv")
    df.to_csv(path, index=False)
    print(f"[generate] ✓ Saved {len(df):,} rows → {path}")


def cmd_train():
    print("[train] Starting model training pipeline …")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_model", os.path.join(ROOT, "src", "train_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def cmd_predict():
    print("[predict] Interactive CLI predictor")
    print("  Type text and press Enter to classify. Type 'quit' to exit.\n")
    try:
        from src.predictor import SentimentPredictor
        p = SentimentPredictor()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Model: {p.model_info().get('model_name','?')}  "
          f"Accuracy: {p.model_info().get('accuracy',0):.2%}\n")

    while True:
        try:
            text = input("  Enter text > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break
        if not text:
            continue
        if text.lower() in ("quit","exit","q"):
            break
        r = p.predict(text)
        print(f"  {r['emoji']}  {r['label'].upper():10}  confidence={r['confidence']:.2%}")
        print(f"     Scores → pos={r['scores'].get('positive',0):.2%}  "
              f"neu={r['scores'].get('neutral',0):.2%}  "
              f"neg={r['scores'].get('negative',0):.2%}\n")


def cmd_evaluate():
    print("[evaluate] Loading model and running test-set evaluation …")
    try:
        from src.predictor import SentimentPredictor
        import pandas as pd
        from sklearn.metrics import classification_report, accuracy_score
        from src.cleaner import clean_series

        p    = SentimentPredictor()
        path = os.path.join(ROOT, "data", "social_media_posts.csv")
        df   = pd.read_csv(path).dropna(subset=["text","sentiment"])
        df["sentiment"] = df["sentiment"].str.lower().str.strip()
        df   = df[df["sentiment"].isin(["positive","neutral","negative"])]

        sample = df.sample(min(500, len(df)), random_state=99)
        preds  = [p.predict(t)["label"] for t in sample["text"]]
        print(classification_report(sample["sentiment"], preds))
        print(f"Accuracy: {accuracy_score(sample['sentiment'], preds):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")


def cmd_dashboard():
    print("[dashboard] Launching Streamlit dashboard …")
    app_path = os.path.join(ROOT, "app", "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)


def cmd_all():
    print(BANNER)
    print("Running full pipeline …\n")
    cmd_generate()
    print()
    cmd_train()
    print()
    print("[all] ✓ Pipeline complete! Launch dashboard with: python main.py dashboard")


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="Social Media Sentiment Analysis Dashboard CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="dashboard",
                        choices=["generate","train","predict","evaluate","dashboard","all"],
                        help=("generate  – create synthetic dataset\n"
                              "train     – train ML model\n"
                              "predict   – interactive CLI predictor\n"
                              "evaluate  – print metrics\n"
                              "dashboard – launch Streamlit (default)\n"
                              "all       – run full pipeline"))

    args = parser.parse_args()
    dispatch = {
        "generate":  cmd_generate,
        "train":     cmd_train,
        "predict":   cmd_predict,
        "evaluate":  cmd_evaluate,
        "dashboard": cmd_dashboard,
        "all":       cmd_all,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
