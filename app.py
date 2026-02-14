from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from inference import predict_price
import numpy as np
import pandas as pd
import joblib
import sys
import os
import glob  # ✅ ADD THIS IMPORT


#-----------------------------------------------------------------------
# DB
from database.db import get_connection, insert_user_hotel

def merge_users(dataset_users_df):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT user_code, name, age, gender, company, password
        FROM users
        WHERE user_code ~ '^[0-9]+$'
    """)
    db_rows = cur.fetchall()

    cur.close()
    conn.close()

    # Convert DB users to dataframe
    if db_rows:
        db_df = pd.DataFrame(db_rows, columns=[
            "code","name","age","gender","company","password"
        ])
        db_df["code"] = db_df["code"].astype(int)
    else:
        db_df = pd.DataFrame(columns=["code","name","age","gender","company","password"])
    
    # Dataset users
    dataset_df = dataset_users_df.copy()
    dataset_df["code"] = dataset_df["code"].astype(int)

    # Merge + remove duplicates (DB overrides)
    merged = pd.concat([dataset_df, db_df.drop(columns=["password"])], ignore_index=True)
    merged = merged.drop_duplicates(subset=["code"], keep="last")

    # Build credentials
    credentials = {}

    # Dataset users → demo password
    for _, r in dataset_df.iterrows():
      code = int(r["code"])
      credentials[code] = {
        "name": r["name"],
        "password": "password123"
    }

     # 2️⃣ Then load DB users (REAL PASSWORD) → overwrite dataset 
    for code, name, age, gender, company, pwd in db_rows:
        credentials[int(code)] = {
        "name": name,
        "password": pwd   # REAL PASSWORD FROM NEON
    }
        
    available = sorted(merged["code"].tolist())

    return merged, available, credentials
from database.db import get_connection

conn = get_connection()
print("Neon DB Connected Successfully")
conn.close()

#---------------------------------------------------------------------
# Load once globally (FAST)
#---------------------------------------------------------------------
flight_df = pd.read_csv("data/flights.csv")

FROM_OPTIONS = sorted(flight_df["from"].dropna().unique())
TO_OPTIONS = sorted(flight_df["to"].dropna().unique())
FLIGHTTYPE_OPTIONS = sorted(flight_df["flightType"].dropna().unique())
AGENCY_OPTIONS = sorted(flight_df["agency"].dropna().unique())


# ---------------------------------------------------------------------------
# Import recommendation engine (must sit beside this file)
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommendation_engine import HotelRecommendationEngine, load_recommendation_models

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'va-dev-secret-change-in-production')


# ===========================================================================
# MODEL LOADING
# ===========================================================================
# -- Recommendation --
print("Loading recommendation models...")
rec_models = load_recommendation_models('models/recommendation')

recommendation_engine = HotelRecommendationEngine(
    user_hotel_matrix  = rec_models['user_hotel_matrix'],
    user_similarity_df = rec_models['user_similarity'],
    hotel_similarity_df= rec_models['hotel_similarity'],
    hotel_features     = rec_models['hotel_features'],
    complete_df        = rec_models['complete_data'],
    users_df           = rec_models['users_data']
)
print("All models loaded successfully.")

rec_models['users_data'], available_users, user_credentials = merge_users(
    rec_models['users_data']
)

recommendation_engine.users_df = rec_models['users_data']

# -- Derived lookups (computed once at startup) --
available_locations = sorted(rec_models['hotel_features']['location'].unique().tolist())

# CRITICAL FIX: Get user codes as integers
available_users = sorted(rec_models['users_data']['code'].astype(int).unique().tolist())

# Debug: Print available user codes
print("\n" + "="*70)
print("AVAILABLE USER CODES:")
print("="*70)
print(f"Total users: {len(available_users)}")
print(f"First 10 users: {available_users[:10]}")
print(f"User code data type: {type(available_users[0]) if available_users else 'No users'}")
print("="*70 + "\n")

print("CREDENTIALS BUILT:")
print(f"Total credentials: {len(user_credentials)}")
print(f"Sample user codes: {list(user_credentials.keys())[:5]}")
print(f"Credential key type: {type(list(user_credentials.keys())[0]) if user_credentials else 'No credentials'}")
print("="*70 + "\n")


#------------------------------------------------------------------------------------------------
# ===========================================================================
# AUTH HELPERS
# ===========================================================================

def login_required(route_fn):
    """Simple decorator – redirects to /login when session is empty."""
    from functools import wraps
    @wraps(route_fn)
    def wrapper(*a, **kw):
        if 'user_code' not in session:
            return redirect(url_for('login'))
        return route_fn(*a, **kw)
    return wrapper

def get_next_user_code():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
                SELECT MAX(CAST(user_code AS INTEGER)) FROM users
        WHERE user_code ~ '^[0-9]+$'
                  """)
    result = cur.fetchone()[0]
    cur.close()
    conn.close()
    if result is None:
        return 1000 
    return result + 1

# ===========================================================================
# AUTH ROUTES
# ===========================================================================

@app.route("/")
def home():
    """Landing page - redirect to login if not authenticated, else dashboard"""
    if 'user_code' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))  # Changed to redirect to login

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user_code' in session and 'created_user_code' not in session:
        return redirect(url_for('dashboard'))

    if request.method == "POST":
        login_type = request.form.get("login_type", "existing")
        
        # EXISTING USER LOGIN
        if login_type == "existing":
            try:
                code = int(request.form.get("user_code", "").strip())
                pwd  = request.form.get("password",  "").strip()
            except ValueError:
                return render_template("login.html",
                                       users=available_users,
                                       error="Invalid user code format.")

            if code in user_credentials and user_credentials[code]['password'] == pwd:
                session['user_code'] = code
                session['user_name'] = user_credentials[code]['name']
                print(f"\nLOGIN SUCCESS: User {code} ({type(code)}) - {user_credentials[code]['name']}")
                return redirect(url_for('dashboard'))

            return render_template("login.html",
                                   users=available_users,
                                   error="Invalid credentials. Please try again.")
        
        # NEW USER REGISTRATION
        elif login_type == "new":
            try:
                # Get form data
                name = request.form.get("new_name", "").strip()
                age = int(request.form.get("new_age", 0))
                gender = request.form.get("new_gender", "").strip()
                company = request.form.get("new_company", "Unknown").strip()
                password = request.form.get("new_password", "").strip()
                
                # Validate
                if not name or age < 18 or age > 100:
                    return render_template("login.html",
                                         users=available_users,
                                         error="Please provide valid registration details.")
                
                if gender not in ['male', 'female', 'unknown']:
                    return render_template("login.html",
                                         users=available_users,
                                         error="Please select a valid gender.")
                
                if len(password) < 6:
                    return render_template("login.html",
                                         users=available_users,
                                         error="Password must be at least 6 characters.")
                
                # Generate new user code
                new_code = get_next_user_code()

                #----------------------INSERT INTO NEON DATABASE-------------
                conn = get_connection()
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO users (user_code, name, age, gender, company, password)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (
                    new_code,
                    name,
                    age,
                    gender,
                    company,
                    password
                ))

                conn.commit()
                cur.close()
                conn.close()

                # -------- UPDATE LOCAL MEMORY (FOR RECOMMENDER) --------
                             
                # Add to credentials
                user_credentials[new_code] = {
                    'password': password,
                    'name': name
                }
                
                # Add to available users list
                available_users.append(new_code)
                available_users.sort()
                
                # Add to users dataframe - ✅ Only if users_data exists
                if 'users_data' in rec_models and rec_models['users_data'] is not None:
                    new_user_row = pd.DataFrame({
                        'code': [new_code],
                        'name': [name],
                        'age': [age],
                        'gender': [gender],
                        'company': [company]
                    })

                    rec_models['users_data'] = pd.concat([rec_models['users_data'], new_user_row], ignore_index=True)
                    recommendation_engine.users_df = rec_models['users_data']
                
                # Create session
                session['user_code'] = new_code
                session['user_name'] = name
                session['is_new_user'] = True
                
                print(f"\nNEW USER REGISTERED: {new_code} - {name} (age: {age}, gender: {gender})")
                #
                session['created_user_code'] = new_code
                #
                return redirect(url_for('login'))
                
            except Exception as e:
              import traceback
              traceback.print_exc()
              return render_template(
               "login.html",
                users=available_users,
                error=str(e)   # show real error temporarily
    )
            
    created_code = session.pop('created_user_code', None)

    return render_template(
            "login.html",
         users=available_users,
         created_code=created_code
              )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

# ===========================================================================
# DASHBOARD
# ===========================================================================

@app.route("/dashboard")
@login_required
def dashboard():
    # Show welcome message for new users
    is_new = session.pop('is_new_user', False)
    return render_template("dashboard.html", 
                         user_name=session['user_name'],
                         user_code=session['user_code'],
                         is_new_user=is_new)

# ===========================================================================
# FLIGHT PREDICTION
# ===========================================================================
@app.route("/flight")
@login_required
def flight():
    return render_template(
        "flight.html",
        user_name=session.get('user_name','User'),
        user_code=session.get('user_code',''),
        from_list=FROM_OPTIONS,
        to_list=TO_OPTIONS,
        flighttype_list=FLIGHTTYPE_OPTIONS,
        agency_list=AGENCY_OPTIONS
    )

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    if request.method == "GET":
        return redirect(url_for("flight"))
    try:
        input_data = {
          "from": request.form["from"],   
          "to": request.form["to"],
          "flightType": request.form["flightType"],
          "agency": request.form["agency"],
          "time": float(request.form["time"]),
          "distance": float(request.form["distance"]),
          "date": request.form["date"]
}

        # 🚫 same origin & destination check
        if input_data["from"] == input_data["to"]:
            raise ValueError("Origin and Destination cannot be same")

        price = predict_price(input_data)

        return render_template(
            "flight.html",
            user_name=session.get('user_name','User'),
            user_code=session.get('user_code',''),
            prediction=f"$ {round(float(price),2)}",
            form_data=input_data,
            from_list=FROM_OPTIONS,
            to_list=TO_OPTIONS,
            flighttype_list=FLIGHTTYPE_OPTIONS,
            agency_list=AGENCY_OPTIONS
        )

    except Exception as e:
        return render_template(
            "flight.html",
            user_name=session.get('user_name','User'),
            user_code=session.get('user_code',''),
            from_list=FROM_OPTIONS,
            to_list=TO_OPTIONS,
            flighttype_list=FLIGHTTYPE_OPTIONS,
            agency_list=AGENCY_OPTIONS,
            prediction="Route not Available, Explore other options ! ",
            form_data=request.form
            
        )
    
@app.route("/get_route_info")
@login_required
def get_route_info():
    from_city = request.args.get("from")
    to_city = request.args.get("to")

    # 🚫 Prevent same route like A → A
    if from_city == to_city:
        return {"error": "Origin and Destination cannot be same"}

    route = flight_df[
        (flight_df["from"] == from_city) &
        (flight_df["to"] == to_city)
    ]

    if not route.empty:
        return {
            "time": float(route.iloc[0]["time"]),
            "distance": float(route.iloc[0]["distance"])
        }
    # ❌ Route not found
    return {"error": "Route not available"}
    


# ===========================================================================
# HOTEL RECOMMENDATIONS
# ===========================================================================

@app.route("/recommend")
@login_required
def recommend():
    return render_template("recommend.html",
                           current_user=session['user_code'],
                           user_name=session.get('user_name', 'User'),
                           locations=available_locations)

@app.route("/get_recommendation", methods=["POST"])
@login_required
def get_recommendation():
    # CRITICAL FIX: Session already stores integer
    user_code = session['user_code']
    
    # Debug logging
    print("\n" + "="*70)
    print("RECOMMENDATION REQUEST DEBUG:")
    print(f"Session user_code: {user_code} (type: {type(user_code)})")
    if 'users_data' in rec_models and rec_models['users_data'] is not None:
        print(f"Available in users_df: {user_code in rec_models['users_data']['code'].values}")
    if 'user_hotel_matrix' in rec_models and rec_models['user_hotel_matrix'] is not None:
        print(f"Available in matrix: {user_code in rec_models['user_hotel_matrix'].index}")
    print("="*70 + "\n")

    try:
        destination      = request.form.get("destination", "").strip()
        budget_min       = float(request.form.get("budget_min", 60))
        budget_max       = float(request.form.get("budget_max", 313))
        n_recommendations = int(request.form.get("n_recommendations", 10))

        # Additional debug
        print(f"Request params: dest={destination}, budget={budget_min}-{budget_max}, n={n_recommendations}")

        # -- validate --
        recommendation_engine.validate_inputs(
            user_code   = user_code,
            destination = destination or None,
            budget_min  = budget_min,
            budget_max  = budget_max
        )

        user_info  = recommendation_engine.get_user_info(user_code)
        user_stats = recommendation_engine.get_user_stats(user_code)
        
        print(f"User info retrieved: {user_info is not None}")
        print(f"User stats: {user_stats}")

        # FIXED: Always enable diversity and adaptive weights (no checkboxes needed)
        recommendations = recommendation_engine.hybrid_recommendations(
            user_code        = user_code,
            destination      = destination or None,
            budget_min       = budget_min,
            budget_max       = budget_max,
            n_recommendations = n_recommendations,
            apply_diversity  = True,  # Always enabled
            use_adaptive_weights = True,  # Always enabled
            debug            = True  # Enable debug mode
        )
        
        print(f"Recommendations generated: {len(recommendations)}")
        if recommendations:
            print("Sample recommendation:")
            print(recommendations[0])
        print("="*70 + "\n")

        return render_template("recommend.html",
                               current_user  = user_code,
                               user_name     = session.get('user_name', 'User'),
                               locations     = available_locations,
                               user_info     = user_info,
                               user_stats    = user_stats,
                               recommendations = recommendations,
                               search_params = {
                                   'destination':      destination,
                                   'budget_min':       budget_min,
                                   'budget_max':       budget_max,
                                   'n_recommendations': n_recommendations
                               })

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return render_template("recommend.html",
                               current_user=user_code,
                               user_name=session.get('user_name', 'User'),
                               locations=available_locations,
                               error=str(ve))
    except Exception as e:
        import traceback
        print("="*70)
        print("EXCEPTION IN get_recommendation:")
        traceback.print_exc()
        print("="*70)
        return render_template("recommend.html",
                               current_user=user_code,
                               user_name=session.get('user_name', 'User'),
                               locations=available_locations,
                               error="Error generating recommendations. Please try again.")

# ===========================================================================
# JSON API ENDPOINTS  (kept for potential AJAX / mobile clients)
# ===========================================================================

@app.route("/api/user_info/<int:user_code>")
@login_required
def get_user_info_api(user_code):
    try:
        info  = recommendation_engine.get_user_info(user_code)
        stats = recommendation_engine.get_user_stats(user_code)
        if info:
            return jsonify(success=True, user_info=info, user_stats=stats)
        return jsonify(success=False, error="User not found"), 404
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route("/api/recommendations", methods=["POST"])
@login_required
def get_recommendations_api():
    try:
        data = request.get_json()
        user_code = session['user_code']

        recommendation_engine.validate_inputs(
            user_code   = user_code,
            destination = data.get('destination'),
            budget_min  = float(data.get('budget_min', 0)),
            budget_max  = float(data.get('budget_max', 10000))
        )

        recs = recommendation_engine.hybrid_recommendations(
            user_code        = user_code,
            destination      = data.get('destination'),
            budget_min       = float(data.get('budget_min', 0)),
            budget_max       = float(data.get('budget_max', 10000)),
            n_recommendations = int(data.get('n_recommendations', 10)),
            apply_diversity  = True,  # Always enabled
            use_adaptive_weights = True  # Always enabled
        )

        return jsonify(
            success=True,
            user_info  = recommendation_engine.get_user_info(user_code),
            user_stats = recommendation_engine.get_user_stats(user_code),
            recommendations = recs
        )
    except ValueError as ve:
        return jsonify(success=False, error=str(ve)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route("/api/locations")
def get_locations_api():
    return jsonify(success=True, locations=available_locations)

@app.route('/health')
def health():
    return {"status": "ok"}, 200
# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')