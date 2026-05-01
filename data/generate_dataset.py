"""
generate_dataset.py
-------------------
Generates a realistic synthetic social media dataset for sentiment analysis.
Run this once to create data/social_media_posts.csv
"""

import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

# ── Positive post templates ──────────────────────────────────────────────────
POSITIVE = [
    "Absolutely love {brand}! The {product} exceeded all my expectations. Highly recommend!",
    "Just received my order from {brand} - delivery was super fast and packaging was perfect!",
    "Been using {brand} for {months} months now and it keeps getting better. Amazing quality!",
    "{brand}'s customer support team is outstanding! Resolved my issue in minutes 🙌",
    "The new {product} from {brand} is a total game changer. Worth every rupee!",
    "Wow! {brand} just launched the most intuitive feature update. Love how they listen to users.",
    "5 stars for {brand}! Consistent quality, great pricing, and wonderful service.",
    "Switched to {brand} last year and never looked back. Phenomenal experience every time.",
    "{brand} is the best in the market hands down. The {product} is flawless!",
    "Placing an order with {brand} is always a pleasure - easy UI, quick checkout, fast delivery!",
    "Can't stop recommending {brand} to everyone I know. Their {product} is pure excellence!",
    "Impressed by {brand}'s commitment to quality. Every product feels premium.",
    "The app by {brand} is so smooth and clean. Best {product} experience I've had!",
    "Shoutout to {brand} for the incredible {product}! My daily life is so much easier now.",
    "Just renewed my {brand} subscription - totally worth it, won't be switching anytime soon.",
]

# ── Negative post templates ──────────────────────────────────────────────────
NEGATIVE = [
    "{brand}'s {product} stopped working after just {days} days. Very disappointed!",
    "Worst customer service experience ever with {brand}. They ignored my complaint for weeks.",
    "Overpriced and underdelivered. {brand}'s {product} is not worth the money at all.",
    "The {brand} app keeps crashing constantly. How is this even released? Unacceptable.",
    "Ordered from {brand} {days} days ago - still no delivery, no update, no response.",
    "Hidden charges in {brand}'s billing again. This is unethical business practice!",
    "Disappointed with {brand}. Product quality has gone downhill drastically this year.",
    "{brand} promised a refund {days} days ago. Still waiting. Very unprofessional.",
    "Never buying from {brand} again. The {product} fell apart within a week. Terrible!",
    "Support team at {brand} is useless. Waited 3 hours to be told they can't help me.",
    "{brand}'s latest update completely broke my workflow. Tons of bugs, no fixes in sight.",
    "Scam! {brand} charged me double and refuses to acknowledge the error. Stay away!",
    "My {product} from {brand} arrived damaged and customer care is unresponsive. Terrible.",
    "How does {brand} stay in business with this kind of quality? Absolute trash.",
    "Regret purchasing the {brand} {product}. Every single feature is half-baked.",
]

# ── Neutral post templates ───────────────────────────────────────────────────
NEUTRAL = [
    "Just tried {brand}'s {product}. It's okay I guess, does what it says it does.",
    "{brand} delivered on time. Product is average - nothing outstanding.",
    "Using {brand} for a while now. It works, but there's room for improvement.",
    "Got the {brand} {product}. Decent for the price range, not exceptional.",
    "Switched from another brand to {brand}. So far it seems comparable.",
    "{brand} updated their app today. Some things changed, some stayed the same.",
    "The {brand} service is functional. Meets basic requirements.",
    "Not sure how I feel about {brand}'s new pricing. Need to think about it.",
    "Tried {brand}'s {product} for the first time. Neither impressed nor disappointed.",
    "{brand} is an option worth considering if you're in the market for {product}.",
    "Average experience with {brand}. Does what it promises, nothing more.",
    "Been with {brand} for {months} months. Stable product, no major complaints or praise.",
    "The {brand} interface is clean enough. Could use some more customization options.",
    "Comparing {brand} with competitors right now. Hard to say who's clearly better.",
    "{brand} support responded in {days} days. Issue half-resolved. Ongoing.",
]

BRANDS   = ["Zomato", "Swiggy", "Flipkart", "Amazon", "Netflix", "Ola", "Meesho",
            "PhonePe", "CRED", "Myntra", "Razorpay", "Zepto"]
PRODUCTS = ["app", "service", "delivery", "subscription", "platform", "product",
            "feature update", "premium plan", "mobile app", "customer support"]
PLATFORMS = ["Twitter", "Instagram", "Reddit", "YouTube"]
CATEGORIES = ["product", "service", "pricing", "support", "delivery"]
TOPICS   = ["Product Quality", "Customer Support", "Delivery Speed", "Pricing",
            "App Experience", "New Feature", "Billing Issue", "UI/UX", "Subscription"]

def gen_row(sent_label, template, idx, base_date):
    brand   = random.choice(BRANDS)
    product = random.choice(PRODUCTS)
    months  = random.randint(1, 18)
    days    = random.randint(2, 14)

    text = template.format(brand=brand, product=product, months=months, days=days)
    ts   = base_date - timedelta(
        days=random.randint(0, 89),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    likes    = int(abs(random.gauss(200, 300))) if sent_label == "positive" else int(abs(random.gauss(80, 150)))
    retweets = int(likes * random.uniform(0.05, 0.25))
    replies  = int(likes * random.uniform(0.02, 0.15))

    return {
        "post_id":   f"POST_{idx:05d}",
        "text":      text,
        "sentiment": sent_label,
        "platform":  random.choice(PLATFORMS),
        "brand":     brand,
        "category":  random.choice(CATEGORIES),
        "topic":     random.choice(TOPICS),
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "likes":     likes,
        "retweets":  retweets,
        "replies":   replies,
        "char_count": len(text),
        "word_count": len(text.split()),
    }

def generate(n_pos=1600, n_neg=700, n_neu=800, seed=42):
    random.seed(seed)
    base = datetime(2025, 4, 1)
    rows = []
    idx  = 1

    for _ in range(n_pos):
        rows.append(gen_row("positive", random.choice(POSITIVE), idx, base)); idx += 1
    for _ in range(n_neg):
        rows.append(gen_row("negative", random.choice(NEGATIVE), idx, base)); idx += 1
    for _ in range(n_neu):
        rows.append(gen_row("neutral",  random.choice(NEUTRAL),  idx, base)); idx += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("social_media_posts.csv", index=False)
    print(f"Dataset saved → social_media_posts.csv")
    print(f"Total rows : {len(df)}")
    print(df["sentiment"].value_counts())
    print(df.head(3))
