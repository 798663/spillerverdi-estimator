import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spillerverdi-estimator", page_icon="⚽")

KATEGORIER = ["position", "sub_position", "foot", "league"]
NUMERISKE = [
    "age", "height_in_cm", "contract_years_left", "contract_missing",
    "games", "minutes", "goals", "assists", "yellow",
    "goals_per90", "assists_per90", "cl_minutes",
    "international_caps", "international_goals",
]


@st.cache_resource
def last_modell():
    return joblib.load("model/model.joblib")


@st.cache_data
def last_spillere():
    df = pd.read_csv("model/spillere.csv")
    df["visningsnavn"] = df["name"] + " (" + df["current_club_name"].fillna("ukjent klubb") + ")"
    return df


def euro(x):
    if x >= 1e6:
        return f"{x / 1e6:.1f} mill. €"
    return f"{x / 1e3:.0f} 000 €"


modell = last_modell()
spillere = last_spillere()

st.title("⚽ Spillerverdi-estimator")
st.caption(
    "Estimerer markedsverdi ut fra sesongen 2025/26. "
    "Modell: HistGradientBoosting, typisk feil ca. 42 %."
)

fane1, fane2 = st.tabs(["Estimer verdi", "Sjekk en ekte spiller"])

# ---------- Fane 1: estimer verdi fra egne tall ----------
with fane1:
    col1, col2 = st.columns(2)
    with col1:
        alder = st.slider("Alder", 16, 40, 24)
        posisjon = st.selectbox("Posisjon", sorted(spillere["position"].unique()))
        liga = st.selectbox("Liga", sorted(spillere["league"].unique()))
        kontrakt = st.slider("År igjen av kontrakten", 0.0, 6.0, 2.0, 0.5)
    with col2:
        minutter = st.number_input("Minutter spilt", min_value=450, max_value=6000, value=1800, step=90)
        mal = st.number_input("Mål", min_value=0, max_value=60, value=5)
        assist = st.number_input("Assist", min_value=0, max_value=40, value=3)

    lignende = spillere[spillere["position"] == posisjon]
    rad = lignende[NUMERISKE].median().to_dict()
    rad.update({
        "age": alder,
        "contract_years_left": kontrakt,
        "contract_missing": 0,
        "minutes": minutter,
        "games": round(minutter * (lignende["games"] / lignende["minutes"]).median()),
        "goals": mal,
        "assists": assist,
        "goals_per90": mal / minutter * 90,
        "assists_per90": assist / minutter * 90,
        "position": posisjon,
        "sub_position": lignende["sub_position"].mode()[0],
        "foot": lignende["foot"].mode()[0],
        "league": liga,
    })

    X = pd.DataFrame([rad])[KATEGORIER + NUMERISKE]
    estimat = modell.predict(X)[0]

    st.metric("Estimert markedsverdi", euro(estimat))
    st.caption(
        "Øvrige egenskaper (høyde, fot, landskamper, CL-minutter, gule kort) "
        "er satt til typisk verdi for posisjonen."
    )

# ---------- Fane 2: sammenlign med en ekte spiller ----------
with fane2:
    valg = st.selectbox(
        "Velg spiller",
        sorted(spillere["visningsnavn"]),
        index=None,
        placeholder="Skriv et navn …",
    )

    if valg:
        s = spillere[spillere["visningsnavn"] == valg].iloc[0]
        faktisk = s["market_value_in_eur"]
        pred = s["predikert"]
        ratio = pred / faktisk

        c1, c2, c3 = st.columns(3)
        c1.metric("Transfermarkt", euro(faktisk))
        c2.metric("Modellen", euro(pred))
        c3.metric("Forhold", f"{ratio:.1f}×")

        if ratio > 1.5:
            st.success("Modellen mener spilleren er **undervurdert**.")
        elif ratio < 0.67:
            st.warning("Modellen mener spilleren er **overvurdert**.")
        else:
            st.info("Modellen er **omtrent enig** med Transfermarkt.")

        st.write(
            f"{s['age']:.0f} år · {s['position']} · {s['league']} · "
            f"{s['minutes']:.0f} min · {s['goals']:.0f} mål · {s['assists']:.0f} assist · "
            f"{s['contract_years_left']:.1f} år igjen av kontrakten"
        )
        st.caption(
            "Modellens anslag er laget uten at modellen har sett denne spilleren "
            "(5-fold kryssvalidering). Transfermarkt-verdier er eksperters vurderinger, "
            "ikke faktiske overgangssummer."
        )