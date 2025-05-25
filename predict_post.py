import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import math
from tensorflow.keras.models import load_model

model = load_model("model_balanced.keras")
preprocessor = joblib.load("preprocessor_balanced.joblib")

desc = input("Enter your post description: ")
platform = input("Platform (e.g. Instagram, Twitter, TikTok): ")
content_type = input("Content type (e.g. Travel, Fitness, Food): ")
hashtags = input("Hashtags (e.g. #fyp, #foodie, #sunset): ")
follower_count = int(input("How many followers does the account have? "))

post_length = len(desc)
hashtag_count = hashtags.count('#')
log_followers = np.log1p(follower_count)
quality_per_follower = post_length / (follower_count + 1)

def compute_score(text):
    hooks = ['best', 'amazing', 'must', 'crazy', 'hidden', 'omg', '🔥', '💯', '😍']
    if isinstance(text, str):
        words = text.lower().split()
        return int(len(words) > 5 or any(h in text.lower() for h in hooks))
    return 0

desc_score = compute_score(desc)

user_df = pd.DataFrame([{
    'Description': desc,
    'Platform': platform,
    'Content_Type': content_type,
    'Post_Length': post_length,
    'Hashtag_Count': hashtag_count,
    'Log_Followers': log_followers,
    'Desc_Quality_Per_Follower': quality_per_follower,
    'Description_Score': desc_score
}])

X_user = preprocessor.transform(user_df)
pred = model.predict(X_user)[0]

likes = max(0, math.ceil(pred[0] * follower_count))
shares = max(0, math.ceil(pred[1] * follower_count))
comments = max(0, math.ceil(pred[2] * follower_count))
total = likes + shares + comments
virality_score = min((total / follower_count) * 100, 100)

def score_to_tier(score):
    if score < 10: return "Low"
    elif score < 30: return "Mediocre"
    elif score < 60: return "Good"
    else: return "Best"

tier = score_to_tier(virality_score)

print("\n📊 Predicted Engagement:")
print(f"Estimated Likes: {likes}")
print(f"Estimated Comments: {comments}")
print(f"Estimated Shares: {shares}")
print(f"Virality Score: {virality_score:.2f}/100 → {tier}")

def plot_virality_meter(score):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)

    zones = [
        (0, 10, 'red'),
        (10, 30, 'orange'),
        (30, 60, 'yellow'),
        (60, 100, 'green')
    ]

    for start, end, color in zones:
        theta = np.linspace(np.pi * (1 - end / 100), np.pi * (1 - start / 100), 50)
        x = np.concatenate(([0], np.cos(theta), [0]))
        y = np.concatenate(([0], np.sin(theta), [0]))
        ax.fill(x, y, color=color, alpha=0.6)

    outline_theta = np.linspace(np.pi, 0, 100)
    ax.plot(np.cos(outline_theta), np.sin(outline_theta), lw=3, color='black', zorder=5)

    angle = (score / 100) * np.pi
    ax.arrow(0, 0, 0.9 * np.cos(np.pi - angle), 0.9 * np.sin(np.pi - angle),
             width=0.02, head_width=0.06, head_length=0.1, color='black', zorder=6)

    ax.text(0, -0.15, f"Virality Score: {score:.1f}/100", fontsize=12, ha='center')
    ax.set_title("Virality Meter", fontsize=14)
    plt.show()

plot_virality_meter(virality_score)
