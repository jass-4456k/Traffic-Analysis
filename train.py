import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model():
    print("Loading data...")
    df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

    df = df[df['temp'] > 200]

    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df = df.drop(columns=['date_time', 'weather_description', 'holiday'])

    df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)

    X = df.drop(columns=['traffic_volume'])
    y = df['traffic_volume']

    feature_names = list(X.columns)
    joblib.dump(feature_names, 'feature_names.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scale_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    joblib.dump(scaler, 'scaler.pkl')

    print("Training Random Forest Regressor (this may take a minute)...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n--- Model Performance ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.4f}")

    joblib.dump(model, 'traffic_model.pkl')
    print("\n✅ Success! Artifacts saved: traffic_model.pkl, scaler.pkl, feature_names.pkl")

if __name__ == "__main__":
    train_model()