import streamlit as st

from utils import load_data
from utils import prepare_data
from utils import produce_confusion
from utils import produce_roc
from utils import round_p
from utils import train_model

st.set_page_config(page_title="Spotify ML", layout="wide")
st.title("Spotify: Predict in Spotify chart")

df, y = load_data()
X_train, X_test, y_train, y_test = prepare_data(df, y)

with st.expander("Data preview"):
    st.dataframe(df.head(15))

#######
# TUTORIAL -
# CREATE THE INPUTS FOR EACH HYPERPARAMETER
#######

with st.sidebar.form(key="hyperparameters_form"):
    st.header("Model Configuration")

    ###### Widgets in here won't rerun the app at every interaction
    random_state = st.slider("Random state",min_value=1,step=1)
    criterion = st.selectbox("Criterion",["gini"])
    c1,c2 = st.columns(2)
    with c1:
        n_estim = st.slider("N of estimators",min_value=1,step=1,max_value=100)
    with c2:
        max_depth = st.slider("Max depth",min_value=1,step=1,max_value=100)
    c1,c2 = st.columns(2)
    with c1:
        min_split = st.slider("Min samples split",min_value=1,step=1,max_value=100)
    with c2:
        min_leaf = st.slider("Min samples leaf",min_value=1,step=1,max_value=100)

    bots = st.checkbox("Use bootstrap?")

    submit_button = st.form_submit_button("Click here to run model", type="primary")

if submit_button:
    hyperparameters = {
        "random_state": random_state,
        "criterion": criterion,
        "n_estimators": n_estim,
        "max_depth": max_depth,
        "min_samples_split": min_split,
        "min_samples_leaf": min_leaf,
        "max_features": 25,
        "bootstrap": bots,
        "n_jobs": -1,
        "max_samples": 0.8,
    }
    (
        train_score,
        test_score,
        precision,
        recall,
        f1,
        confusion,
        seconds_run,
        fpr,
        tpr,
        roc_auc,
    ) = train_model(hyperparameters, X_train, X_test, y_train, y_test)

    st.write(f"Model ran in: {round(seconds_run,4)} seconds")
    c1,c2 = st.columns(2)
    with c1:
        st.metric(label="Training Score", value=round_p(train_score))
    with c2:
        st.metric(
            label="Test Score",
            value=round_p(test_score),
            delta=round_p(test_score - train_score),
        )
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric(label="Precision", value=round_p(precision))
    with c2:
        st.metric(label="Recall", value=round_p(recall))
    with c3:
        st.metric(label="F1", value=round_p(f1))

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(produce_confusion(confusion), use_container_width=True)
    with c2:
        st.altair_chart(produce_roc(fpr, tpr, roc_auc), use_container_width=True)
