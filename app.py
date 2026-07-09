import streamlit as st
from src.styles import load_css
from src.github_api import fetch_issue
from src.feature_builder import build_features
from src.predictor import predict_resolution_time

st.set_page_config(
    page_title="GitHub Issue Resolution Predictor",
    page_icon=":material/code:",
    layout="wide"
)

load_css()

st.title(":material/code: GitHub Issue Resolution Predictor")
st.markdown("""
Predict the **expected resolution time** of a GitHub issue using a Machine Learning model trained on real GitHub issue history.

Paste the URL of any **public GitHub issue** below.
""")

st.divider()

with st.form("issue_form"):
    issue_url = st.text_input(
        "🔗 GitHub Issue URL",
        placeholder="https://github.com/microsoft/vscode/issues/251455",
    )
    submitted = st.form_submit_button(
        ":material/rocket_launch: Analyze Issue",
        use_container_width=True
    )

if submitted:
    if not issue_url.strip():
        st.warning("Please enter a GitHub issue URL.")
    else:
        with st.spinner("Fetching issue from GitHub..."):
            try:
                issue = fetch_issue(issue_url)
                features = build_features(issue)
                prediction = predict_resolution_time(features)

                st.session_state["issue"] = issue
                st.session_state["prediction"] = prediction

            except Exception as e:
                st.error(str(e))

if "issue" in st.session_state:

    issue = st.session_state["issue"]
    prediction = st.session_state["prediction"]

    repo = issue["repository"]
    labels = issue.get("labels", [])

    st.success("Issue analyzed successfully!")

    left_pred, right_pred = st.columns([1.2,1])

    with left_pred:

        st.subheader("🤖 ML Prediction")

        days = prediction["days"]
        hours = prediction["hours"]

        st.metric(
            "Estimated Resolution Time",
            f"{days:.2f} Days",
            f"{hours:.1f} Hours"
        )

        st.progress(min(days/30,1.0))

        if days < 1:
            st.success(":material/rocket_launch: Likely to be resolved within a day.")
        elif days < 7:
            st.info("📅 Expected within one week.")
        elif days < 30:
            st.warning("⏳ May require several weeks.")
        else:
            st.error("🐢 May take more than a month.")

    with right_pred:

        st.subheader("📦 Repository")

        c1,c2 = st.columns(2)

        with c1:
            st.metric("⭐ Stars", f"{repo.get('stargazers_count',0):,}")
            st.metric("🐞 Open Issues", f"{repo.get('open_issues_count',0):,}")

        with c2:
            st.metric("🍴 Forks", f"{repo.get('forks_count',0):,}")
            st.metric("👀 Watchers", f"{repo.get('watchers_count',0):,}")

        st.write(f"**Repository:** {repo['full_name']}")
        st.write(f"**Language:** {repo.get('language','Unknown')}")

        if repo.get("description"):
            st.info(repo["description"])

    st.divider()

    st.subheader("🐞 Issue Details")

    st.markdown(f"### {issue['title']}")

    left,right = st.columns([2,1])

    with left:
        st.write(f"**State:** `{issue['state']}`")
        st.write(f"**Author Association:** `{issue['author_association']}`")
        st.write(f"**Comments:** {issue['comments']}")

        st.write("### Labels")

        if labels:
            cols = st.columns(min(4,len(labels)))
            for i,label in enumerate(labels):
                cols[i % len(cols)].markdown(
                    f"""
                    <div style="
                    background-color:#{label['color']};
                    padding:8px;
                    border-radius:10px;
                    text-align:center;
                    color:white;
                    font-weight:bold;
                    margin-bottom:8px;">
                    {label['name']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.write("No Labels")

    with right:
        st.write("### Created")
        st.write(issue["created_at"][:10])

        st.write("### Updated")
        st.write(issue["updated_at"][:10])

        st.write("### Repository")
        st.write(repo["full_name"])

    st.divider()

    st.caption(
        "Built with ❤️ using Python • Streamlit • XGBoost • Scikit-learn • GitHub REST API"
    )
