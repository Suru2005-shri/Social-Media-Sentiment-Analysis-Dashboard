"""
src/train_model.py
Run: python src/train_model.py
"""
import os, sys, json, time
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import ComplementNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.cleaner  import clean_series
from src.features import FeatureBuilder

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH=os.path.join(ROOT,"data","social_media_posts.csv")
MODEL_DIR=os.path.join(ROOT,"models")
IMG_DIR  =os.path.join(ROOT,"outputs","charts")
for d in [MODEL_DIR,IMG_DIR]: os.makedirs(d,exist_ok=True)
LABEL_MAP={"positive":2,"neutral":1,"negative":0}
PALETTE  ={"positive":"#22c55e","neutral":"#f59e0b","negative":"#ef4444"}
plt.rcParams.update({"figure.facecolor":"#0a0c10","axes.facecolor":"#12151c",
    "axes.edgecolor":"#2a3045","axes.labelcolor":"#8892aa",
    "xtick.color":"#8892aa","ytick.color":"#8892aa","text.color":"#e8eaf0"})

def main():
    print("="*60+"\n  Sentiment Analysis - Model Training\n"+"="*60)
    df=pd.read_csv(DATA_PATH).dropna(subset=["text","sentiment"])
    df["sentiment"]=df["sentiment"].str.lower().str.strip()
    df=df[df["sentiment"].isin(LABEL_MAP)].reset_index(drop=True)
    print(f"Rows: {len(df)}\n{df['sentiment'].value_counts().to_string()}")

    df["clean_text"]=clean_series(df["text"],remove_stops=True,lemmatize=False)
    df["label"]=df["sentiment"].map(LABEL_MAP)
    df=df[df["clean_text"].str.strip()!=""].reset_index(drop=True)

    Xc,Xr,y=df["clean_text"].tolist(),df["text"].tolist(),df["label"].values
    Xc_tr,Xc_te,Xr_tr,Xr_te,y_tr,y_te=train_test_split(Xc,Xr,y,test_size=0.2,random_state=42,stratify=y)
    fb=FeatureBuilder(max_tfidf=5000)
    X_train=fb.fit_transform(Xc_tr,Xr_tr); X_test=fb.transform(Xc_te,Xr_te)
    print(f"\nTrain {X_train.shape}  Test {X_test.shape}  Vocab {fb.vocabulary_size():,}")

    clf=LogisticRegression(C=1.5,max_iter=1000,class_weight="balanced",solver="lbfgs",random_state=42)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv=cross_val_score(clf,X_train,y_tr,cv=skf,scoring="f1_macro",n_jobs=-1)
    print(f"CV F1: {cv.mean():.4f} +/- {cv.std():.4f}")
    clf.fit(X_train,y_tr)

    y_pred=clf.predict(X_test)
    acc=accuracy_score(y_te,y_pred); f1=f1_score(y_te,y_pred,average="macro")
    cm=confusion_matrix(y_te,y_pred,labels=[0,1,2])
    print(classification_report(y_te,y_pred,target_names=["negative","neutral","positive"],labels=[0,1,2]))
    print(f"Accuracy: {acc:.4f}   F1 Macro: {f1:.4f}")

    # Charts
    fig,ax=plt.subplots(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
        xticklabels=["neg","neu","pos"],yticklabels=["neg","neu","pos"],
        ax=ax,annot_kws={"size":13,"color":"#1a1e28"})
    ax.set_title(f"Confusion Matrix (Acc={acc:.2%})"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/confusion_matrix.png",dpi=150,bbox_inches="tight",facecolor="#0a0c10"); plt.close()

    vc=df["sentiment"].value_counts()
    fig,ax=plt.subplots(figsize=(7,4))
    bars=ax.bar(vc.index,vc.values,color=[PALETTE.get(l,"#8892aa") for l in vc.index],edgecolor="#2a3045",width=.55)
    for bar,v in zip(bars,vc.values): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+15,f"{v:,}",ha="center",fontsize=11)
    ax.set_title("Sentiment Distribution"); ax.set_ylabel("Posts"); ax.grid(axis="y",alpha=0.3)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/sentiment_distribution.png",dpi=150,bbox_inches="tight",facecolor="#0a0c10"); plt.close()

    pivot=df.groupby(["platform","sentiment"]).size().unstack(fill_value=0)
    fig,ax=plt.subplots(figsize=(8,4))
    pivot.plot(kind="bar",ax=ax,color=[PALETTE.get(c,"#8892aa") for c in pivot.columns],edgecolor="#2a3045",width=.65)
    ax.set_title("Sentiment by Platform"); ax.set_xlabel(""); ax.set_ylabel("Posts")
    ax.legend(facecolor="#12151c",edgecolor="#2a3045",labelcolor="#e8eaf0"); ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
    ax.grid(axis="y",alpha=0.3); ax.spines[["top","right","left","bottom"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/platform_sentiment.png",dpi=150,bbox_inches="tight",facecolor="#0a0c10"); plt.close()
    print(f"Charts saved to {IMG_DIR}/")

    joblib.dump(clf,f"{MODEL_DIR}/sentiment_model.pkl")
    joblib.dump(fb, f"{MODEL_DIR}/feature_builder.pkl")
    json.dump({"model_name":"Logistic Regression","accuracy":round(acc,4),"f1_macro":round(f1,4),
               "label_map":LABEL_MAP,"trained_at":time.strftime("%Y-%m-%d %H:%M:%S")},
              open(f"{MODEL_DIR}/model_meta.json","w"),indent=2)
    print(f"Model saved to {MODEL_DIR}/")
    print(f"\nDONE! Accuracy={acc:.2%} F1={f1:.4f}")
    print("Launch: streamlit run app/dashboard.py")

if __name__=="__main__": main()
