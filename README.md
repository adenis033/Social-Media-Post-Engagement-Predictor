# Social Media Post Engagement Predictor

This project uses a neural network to predict the engagement of a social media post — including likes, shares, comments, and a virality score — based on the post's content and account metrics. It leverages text processing, follower data, and platform context to provide accurate forecasts.

## 📊 What It Does

- Predicts **Likes**, **Shares**, and **Comments**
- Computes a **Virality Score** (0–100) based on total predicted engagement
- Visualizes results using a **Virality Meter**
- Accepts inputs like post description, platform, hashtags, and follower count

## 🧠 Model Details

- Framework: Keras (TensorFlow backend)
- Layers: Dense(128) → Dense(64) → Dense(3)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- EarlyStopping enabled (patience = 20)

## 🛠️ Preprocessing

- Text: TF-IDF vectorization on post descriptions
- Categorical: One-hot encoding for Platform & Content_Type
- Numeric: Standardized features like Post Length, Hashtag Count, Log Followers, Description Score

## 📁 Files

- `train_model.py` — trains the Keras model on `Cleaned_Engagement_Dataset.csv`, saves:
  - `model_balanced.keras`
  - `preprocessor_balanced.joblib`
  - `training_balanced_plot.png`

- `predict_post.py` — loads model & preprocessor, takes user input, outputs predictions and shows a custom virality gauge.

## ▶️ How to Run

1. **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib tensorflow joblib
    ```

2. **Train the model**:
    ```bash
    python train_model.py
    ```

3. **Run prediction**:
    ```bash
    python predict_post.py
    ```

4. **Follow CLI prompts**:
    - Post description
    - Platform (Instagram, TikTok, etc.)
    - Content type (e.g. Travel, Fitness)
    - Hashtags (e.g. #fyp #explore)
    - Follower count

## 📷 Example Output

````
📊 Predicted Engagement:
Estimated Likes: 834
Estimated Comments: 92
Estimated Shares: 134
Virality Score: 78.3/100 → Best
````

Also includes a visual **Virality Meter** in a pop-up window.

## 👤 Author

**Denis-Andrei Râpa** — [GitHub Profile](https://github.com/adenis033)
