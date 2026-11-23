import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "CreditScore": np.random.randint(350, 850, n),
    "Age": np.random.randint(18, 80, n),
    "Tenure": np.random.randint(0, 10, n),
    "Balance": np.random.uniform(0, 200000, n),
    "NumOfProducts": np.random.randint(1, 5, n),
    "HasCrCard": np.random.randint(0, 2, n),
    "IsActiveMember": np.random.randint(0, 2, n),
    "EstimatedSalary": np.random.uniform(10000, 150000, n),
})

y = ((df["Age"] > 45) & (df["IsActiveMember"] == 0) | (df["NumOfProducts"] >= 3)).astype(int)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(df, y)
joblib.dump(modelo, "modelo_rfchurn_tunning.pkl")
print("¡Modelo guardado con éxito como modelo_rfchurn_tunning.pkl!")
