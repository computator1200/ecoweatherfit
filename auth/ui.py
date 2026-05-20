import streamlit as st
import extra_streamlit_components as stx
from datetime import datetime, timedelta
import config.settings
from auth.service import (
    login, signup, verify_email_token, get_current_user_from_token,
    add_wardrobe_item, list_wardrobe, delete_wardrobe_item,
    list_history
)
from auth.database import get_session
from auth.models import UserPreference

def get_cookie_manager():
    """Return a per-session singleton CookieManager.

    Cannot use ``@st.cache_resource`` here: ``stx.CookieManager()`` creates a
    Streamlit component (a frontend widget), and Streamlit's cache primitives
    will only re-run the wrapped function on a cache miss — which means the
    widget would never render after the first page load, breaking cookie
    reads. The right scope for a per-user component instance is
    ``st.session_state``, which is per-tab and re-runs the widget on every
    script execution. The fixed ``key`` argument lets the component framework
    deduplicate within a single script run.
    """
    if "_cookie_manager" not in st.session_state:
        st.session_state["_cookie_manager"] = stx.CookieManager(
            key="ecoweatherfit_cookie_mgr"
        )
    return st.session_state["_cookie_manager"]

def render_auth_gate() -> int | None:
    # Check for verify_token in query params first
    if "verify_token" in st.query_params:
        render_verification_page()
        return None
        
    cm = get_cookie_manager()
    token = cm.get("ecoweatherfit_session")
    
    if token:
        user = get_current_user_from_token(token)
        if user:
            return user.id
            
    # Not logged in
    st.title("Welcome to EcoWeatherFit")
    tab1, tab2 = st.tabs(["Log In", "Sign Up"])
    
    with tab1:
        render_login_form(cm)
        
    with tab2:
        render_signup_form()
        
    return None

def render_login_form(cm):
    st.subheader("Log In")
    with st.form("login_form"):
        identifier = st.text_input(
            "Email or username",
            help="You can sign in with either the email you signed up with or your username.",
        )
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

        if submitted:
            result = login(identifier, password)
            if result.success:
                # extra-streamlit-components uses snake_case `same_site` and
                # lowercase 'lax'/'strict'. `secure=False` is fine for the
                # academic HTTP demo; set to True behind any HTTPS reverse
                # proxy in production.
                cm.set(
                    "ecoweatherfit_session",
                    result.jwt_token,
                    expires_at=datetime.now() + timedelta(days=7),
                    same_site="lax",
                    secure=False,
                    key="set_session_cookie",
                )
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(result.error)


def render_signup_form():
    st.subheader("Create Account")
    # Reading SIGNUP_CODE at render time (not module import) so changes to the
    # env var take effect on Streamlit's next rerun without a process restart.
    required_code = getattr(config.settings, "SIGNUP_CODE", "") or ""
    code_gated = bool(required_code)

    with st.form("signup_form"):
        email = st.text_input("Email")
        username = st.text_input(
            "Username",
            help="3–32 chars, letters/digits/underscore/dot/hyphen. Displayed on your account.",
        )
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        signup_code_value = ""
        if code_gated:
            signup_code_value = st.text_input(
                "Signup code",
                type="password",
                help="This deployment requires a pre-shared code to create an account. "
                     "Ask the project owner if you don't have one.",
            )
        submitted = st.form_submit_button("Create Account")

        if submitted:
            if password != password_confirm:
                st.error("Passwords do not match.")
            else:
                app_base_url = getattr(config.settings, "APP_BASE_URL", "http://localhost:8501")
                result = signup(
                    email=email,
                    username=username,
                    password=password,
                    app_base_url=app_base_url,
                    signup_code=signup_code_value,
                )
                if result.success:
                    if code_gated:
                        # The signup-code path auto-verifies; user can log in immediately.
                        st.success(
                            f"Account created for **{result.username}**! "
                            "You can log in straight away on the Log In tab."
                        )
                    else:
                        st.success("Account created successfully!")
                        st.info("Please check your email to verify your account.")
                        fallback_link = st.session_state.get("_last_verification_link")
                        if fallback_link:
                            st.warning(
                                "SMTP is not configured on this deployment, so the "
                                "verification link is shown directly below. In a "
                                "production deployment this would be emailed."
                            )
                            st.markdown(f"[{fallback_link}]({fallback_link})")
                else:
                    st.error(result.error)

def render_verification_page():
    st.title("Email Verification")
    token = st.query_params.get("verify_token")
    if not token:
        st.error("No verification token found.")
        return
        
    result = verify_email_token(token)
    if result.success:
        st.success("Email verified successfully! You can now log in.")
    else:
        st.error(result.error or "Failed to verify email.")
        
    if st.button("Go to Login"):
        del st.query_params["verify_token"]
        st.rerun()

def render_user_menu(user_id: int) -> dict | None:
    cm = get_cookie_manager()

    # Pull the user's display name + prefs in a single round-trip.
    from auth.models import User
    with get_session() as session:
        user = session.query(User).filter_by(id=user_id).first()
        pref = session.query(UserPreference).filter_by(user_id=user_id).first()
        if user is None or pref is None:
            return None

        display_name = user.username or user.email.split("@")[0]
        st.sidebar.markdown(
            f"### 👤 Signed in as **{display_name}**\n"
            f"<span style='color:#666;font-size:0.85em;'>{user.email}</span>",
            unsafe_allow_html=True,
        )

        if st.sidebar.button("Log out", key="logout_btn"):
            cm.delete("ecoweatherfit_session", key="del_session_cookie")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        return {
            "default_city": pref.default_city,
            "gender_presentation": pref.gender_presentation,
            "style_preference": pref.style_preference,
            "show_eco_tips": pref.show_eco_tips,
            "show_circular_loop": pref.show_circular_loop
            }
    return None

def render_wardrobe_manager(user_id: int):
    with st.expander("Wardrobe Manager"):
        st.write("Add items to your wardrobe:")
        with st.form("add_wardrobe_item"):
            label = st.text_input("Label (e.g. Blue wool coat)")
            category = st.selectbox("Category", ["coat", "jumper", "shirt", "trousers", "shoes", "accessory", "other"])
            warmth = st.selectbox("Warmth", ["light", "medium", "warm", "heavy"])
            waterproof = st.checkbox("Waterproof")
            notes = st.text_area("Notes (optional)")
            submitted = st.form_submit_button("Add Item")
            
            if submitted:
                if label:
                    res = add_wardrobe_item(user_id, label, category, warmth, waterproof, notes)
                    if res.success:
                        st.success("Item added!")
                        st.rerun()
                else:
                    st.error("Label is required.")
                    
        st.write("Current Wardrobe:")
        items = list_wardrobe(user_id)
        if not items:
            st.info("Your wardrobe is empty.")
        else:
            for item in items:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{item.label}** ({item.category}) - {item.warmth}" + (" - Waterproof" if item.waterproof else ""))
                with col2:
                    if st.button("Delete", key=f"del_item_{item.id}"):
                        delete_wardrobe_item(user_id, item.id)
                        st.rerun()

def render_history_panel(user_id: int):
    with st.expander("Forecast History"):
        history = list_history(user_id)
        if not history:
            st.info("No forecast history yet.")
        else:
            for h in history:
                st.markdown(f"**{h.city}** on {h.requested_at.strftime('%Y-%m-%d %H:%M')}")
                st.write(h.recommendation_text)
                st.divider()
