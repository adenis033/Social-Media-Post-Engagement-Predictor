import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("Cleaned_Engagement_Dataset.csv")

df['Log_Followers'] = np.log1p(df['Followers'])
df['Desc_Quality_Per_Follower'] = df['Post_Length'] / (df['Followers'] + 1)

df['Likes_per_Follower'] = df['Likes_per_Follower'].clip(lower=0.005)
df['Shares_per_Follower'] = df['Shares_per_Follower'].clip(lower=0.001)
df['Comments_per_Follower'] = df['Comments_per_Follower'].clip(lower=0.001)

X_raw = df[['Description', 'Platform', 'Content_Type', 'Post_Length',
            'Hashtag_Count', 'Log_Followers', 'Desc_Quality_Per_Follower', 'Description_Score']]
y = df[['Likes_per_Follower', 'Shares_per_Follower', 'Comments_per_Follower']].values

text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=200)),
    ('scale', StandardScaler(with_mean=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('desc', text_pipeline, 'Description'),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Platform', 'Content_Type']),
    ('num', StandardScaler(), ['Post_Length', 'Hashtag_Count', 'Log_Followers', 'Desc_Quality_Per_Follower', 'Description_Score'])
])

X = preprocessor.fit_transform(X_raw)
joblib.dump(preprocessor, "preprocessor_balanced.joblib")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=250, batch_size=32,
                    callbacks=[early_stop], verbose=1)

model.save("model_balanced.keras")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_balanced_plot.png")
plt.show()
