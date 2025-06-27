import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("📁 CSV-Datei hochladen (mit ; getrennt)", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV einlesen ---
try:
    df = pd.read_csv(uploaded_file, sep=";", decimal=",")
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("🧾 Gefundene Spalten:", df.columns.tolist())

# --- Zielgrößen aus numerischen Spalten dynamisch auswählen ---
numerische_spalten = df.select_dtypes(include=[np.number]).columns.tolist()

if not numerische_spalten:
    st.error("❌ Keine numerischen Spalten im Datensatz gefunden.")
    st.stop()

zielspalten = st.multiselect(
    "🎯 Zielgrößen auswählen (numerische Spalten)",
    options=numerische_spalten,
    default=[numerische_spalten[0]]
)

if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

# --- Eingabe- und Zielvariablen trennen ---
X = df.drop(columns=zielspalten, errors="ignore")
y = df[zielspalten].copy()

# Spaltentypen bestimmen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)

# Fehlende Werte bereinigen
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Daten zum Trainieren.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Benutzer-Eingabeformular ---
st.sidebar.header("🔧 Parameter anpassen")
user_input = {}

for col in numerisch:
    try:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    except:
        continue

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten auffüllen
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]

st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Partial Dependence Plot ---
st.subheader("📊 Einflussanalyse (Partial Dependence)")

feature_options = X_clean.columns.tolist()
selected_features = st.multiselect("📌 Einflussgrößen auswählen (max. 2)", feature_options, default=[feature_options[0]])

if len(selected_features) > 2:
    st.warning("Bitte wähle maximal 2 Einflussgrößen aus.")
    selected_features = selected_features[:2]

selected_targets = st.multiselect("📈 Zielgrößen für Analyse", zielspalten, default=zielspalten[:1])

if selected_features and selected_targets:
    for ziel in selected_targets:
        try:
            target_index = zielspalten.index(ziel)
            fig, ax = plt.subplots(figsize=(6, 4) if len(selected_features) == 1 else (6, 6))
            PartialDependenceDisplay.from_estimator(
                modell, X_clean, selected_features, target=target_index, ax=ax
            )
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ PDP für {ziel} konnte nicht erstellt werden: {e}")

# --- Regelmodul: Einfluss-Komponenten analysieren ---
st.subheader("💬 Regelbasierte Analyse per Auswahl")

regel_basis = {
    ("sylysia256", "glanz20"): "📌 Mehr Sylysia256 → tendenziell geringerer Glanz.",
    ("lackslurry", "kostengesamtkg"): "📌 Höherer Lackslurry-Anteil → höhere Kosten.",
    ("acrysolrm2020e", "brookfield"): "📌 AcrysolRM2020E erhöht tendenziell die Brookfield-Viskosität.",
    ("byk1770", "viskositätlowshear"): "📌 BYK1770 beeinflusst die Viskosität je nach Anteil."
}

komponenten_dropdown = st.selectbox("🧱 Komponente wählen", sorted([c.lower() for c in X.columns]))
ziel_dropdown = st.selectbox("🎯 Zielgröße wählen", sorted([z.lower() for z in zielspalten]))

regel_key = (komponenten_dropdown, ziel_dropdown)
antwort = regel_basis.get(regel_key)

if antwort:
    st.success(antwort)
else:
    st.info("❓ Für diese Kombination ist noch keine Regel hinterlegt.")

# --- Was-wäre-wenn Analyse ---
st.subheader("🔁 Was wäre wenn du eine Komponente änderst?")
was_waere_feature = st.selectbox("🔄 Komponente auswählen", numerisch)

input_varianten = pd.DataFrame([
    {**user_input, was_waere_feature: df[was_waere_feature].min()},
    {**user_input, was_waere_feature: df[was_waere_feature].mean()},
    {**user_input, was_waere_feature: df[was_waere_feature].max()},
])

encoded_varianten = pd.get_dummies(input_varianten)
for col in X_clean.columns:
    if col not in encoded_varianten.columns:
        encoded_varianten[col] = 0
encoded_varianten = encoded_varianten[X_clean.columns]

vorhersagen = modell.predict(encoded_varianten)

st.markdown(f"📊 Vergleich für **{was_waere_feature}**:")
was_labels = ["Minimum", "Mittelwert", "Maximum"]

for i, ziel in enumerate(zielspalten):
    st.write(f"**{ziel}:**")
    st.write({
        was_labels[j]: round(vorhersagen[j][i], 2)
        for j in range(3)
    })

# --- Download der Vorhersage ---
st.subheader("📤 Vorhersage exportieren")

vorhersage_df = pd.DataFrame([prediction], columns=zielspalten)
csv_download = vorhersage_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download als CSV",
    data=csv_download,
    file_name="vorhersage.csv",
    mime="text/csv"
)
