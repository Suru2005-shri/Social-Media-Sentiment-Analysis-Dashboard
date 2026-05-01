"""
src/predictor.py  –  Load trained model and run inference.
Usage: python src/predictor.py
"""
import os, sys, json, joblib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.cleaner  import clean_text
from src.features import vader_scores, vader_label, VADER_OK

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
LABEL_MAP = {2:"positive",1:"neutral",0:"negative"}
EMOJI_MAP = {"positive":"😊","neutral":"😐","negative":"😠"}
COLOR_MAP = {"positive":"#22c55e","neutral":"#f59e0b","negative":"#ef4444"}

class SentimentPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        mp = os.path.join(model_dir,"sentiment_model.pkl")
        fp = os.path.join(model_dir,"feature_builder.pkl")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model not found: {mp}\nRun: python src/train_model.py")
        self.clf  = joblib.load(mp)
        self.fb   = joblib.load(fp)
        meta_path = os.path.join(model_dir,"model_meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    def predict(self, text: str) -> dict:
        clean = clean_text(text, remove_stops=True, lemmatize=False)
        if not clean.strip():
            label = "neutral"
            return {"label":label,"confidence":0.5,"scores":{"positive":0.33,"neutral":0.34,"negative":0.33},
                    "emoji":EMOJI_MAP[label],"color":COLOR_MAP[label],"clean_text":clean,
                    "vader_scores":{"compound":0.0,"pos":0.33,"neu":0.34,"neg":0.33}}
        X     = self.fb.transform([clean],[text])
        pred  = int(self.clf.predict(X)[0])
        proba = self.clf.predict_proba(X)[0]
        label = LABEL_MAP[pred]
        classes   = [LABEL_MAP[int(c)] for c in self.clf.classes_]
        score_d   = dict(zip(classes,[round(float(p),4) for p in proba]))
        conf      = round(float(proba[list(self.clf.classes_).index(pred)]),4)
        vs        = vader_scores(text) if VADER_OK else {"compound":0,"pos":score_d.get("positive",0),"neu":score_d.get("neutral",0),"neg":score_d.get("negative",0)}
        return {"label":label,"confidence":conf,"scores":score_d,
                "emoji":EMOJI_MAP[label],"color":COLOR_MAP[label],
                "clean_text":clean,"vader_scores":vs}

    def predict_batch(self, texts): return [self.predict(t) for t in texts]
    def model_info(self): return self.meta

if __name__=="__main__":
    p = SentimentPredictor()
    print("Model:", p.model_info())
    tests=["Absolutely love this! Best purchase ever.",
           "Terrible experience. Never ordering again.",
           "It is okay, nothing special.",
           "App keeps crashing. Very disappointed!"]
    print(f"\n{'Text':<52} {'Label':<12} {'Conf':>6}")
    print("-"*74)
    for t in tests:
        r=p.predict(t)
        print(f"{t[:51]:<52} {r['emoji']} {r['label']:<10} {r['confidence']:>6.2%}")
