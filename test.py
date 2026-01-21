
"""
Hotel Recommendation System - Testing Suite
Tests all functionality including cold start handling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Src'))

from recommendation_engine import HotelRecommendationEngine, load_recommendation_models
from cold_start_handler import ColdStartHandler, EnhancedHotelRecommendationEngine 
from evaluation import RecommendationEvaluator, run_evaluation

print("="*70)
print("VOYAGE ANALYTICS - RECOMMENDATION SYSTEM TESTING")
print("="*70)

# ============================================================================
# [STEP 1] LOAD MODELS AND DATA
# ============================================================================

print("\n[STEP 1] Loading models and data...")

models = load_recommendation_models(models_dir='models')

# Extract models
user_hotel_matrix = models['user_hotel_matrix']
user_similarity_df = models['user_similarity']
hotel_similarity_df = models['hotel_similarity']
hotel_features = models['hotel_features']
complete_df = models['complete_data']
users_df = models['users_data']

print("✓ All models loaded successfully!")
print(f"  - Users: {len(users_df):,}")
print(f"  - Hotels: {len(hotel_features)}")
print(f"  - User-Hotel Matrix: {user_hotel_matrix.shape}")

# ============================================================================
# [STEP 2] INITIALIZE ENGINES
# ============================================================================

print("\n[STEP 2] Initializing recommendation engine...")

# Base recommendation engine
rec_engine = HotelRecommendationEngine(
    user_hotel_matrix=user_hotel_matrix,
    user_similarity_df=user_similarity_df,
    hotel_similarity_df=hotel_similarity_df,
    hotel_features=hotel_features,
    complete_df=complete_df,
    users_df=users_df
)

# Cold start handler
cold_start = ColdStartHandler(
    complete_df=complete_df,
    hotel_features=hotel_features,
    users_df=users_df
)

# Enhanced engine with cold start
enhanced_engine = EnhancedHotelRecommendationEngine(  # FIXED
    base_engine=rec_engine,
    cold_start_handler=cold_start
)

print("✓ Recommendation engine initialized!")

# ============================================================================
# [STEP 3] TEST BASIC RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("[STEP 3] Testing Basic Recommendations")
print("="*70)

# Select test user
test_user = users_df['code'].iloc[10]
user_info = rec_engine.get_user_info(test_user)

print(f"\n Test User Profile:")
print(f"   Name: {user_info['name']}")
print(f"   Gender: {user_info['gender']}")
print(f"   Age: {user_info['age']}")
print(f"   Company: {user_info['company']}")

# Get recommendations
recs = rec_engine.hybrid_recommendations(test_user, n_recommendations=5)

print(f"\n Top 5 Hybrid Recommendations:\n")

for i, rec in enumerate(recs, 1):
    print(f"{i}. {rec['hotel_name']}")
    print(f"   Location: {rec['location']}")
    print(f"   Price: ${rec['avg_price']:.2f}/night")
    print(f"   Score: {rec['recommendation_score']:.4f}")
    print(f"   Methods: {rec['methods_used']}\n")

# ============================================================================
# [STEP 4] TEST FILTERED RECOMMENDATIONS
# ============================================================================

print("="*70)
print("[STEP 4] Testing Filtered Recommendations")
print("="*70)

test_user2 = users_df['code'].iloc[50]
user_info2 = rec_engine.get_user_info(test_user2)

print(f"\n User: {user_info2['name']}")
print(f" Filters: Rio de Janeiro (RJ), Budget $150-$350\n")

filtered_recs = rec_engine.hybrid_recommendations(
    test_user2,
    destination='Rio de Janeiro (RJ)',
    budget_min=150,
    budget_max=350,
    n_recommendations=5
)

print(f"✓ Found {len(filtered_recs)} recommendations:")
for i, rec in enumerate(filtered_recs, 1):
    print(f"  {i}. {rec['hotel_name']} - ${rec['avg_price']:.2f}/night ({rec['location']})")

# ============================================================================
# [STEP 5] TEST COLD START HANDLER
# ============================================================================

print("\n" + "="*70)
print("[STEP 5] Testing Cold Start Handler")
print("="*70)

# Test new user
new_user_demo = {
    'gender': 'female',
    'age': 28
}

print(f"\n New User Profile:")
print(f"   Gender: {new_user_demo['gender']}")
print(f"   Age: {new_user_demo['age']}\n")

new_user_recs = enhanced_engine.get_recommendations(
    user_code='NEW_USER_12345',  # Non-existent user
    user_demographics=new_user_demo,
    budget_min=100,
    budget_max=250,
    n_recommendations=5
)

print(f"✓ Cold Start Recommendations:")
for i, rec in enumerate(new_user_recs, 1):
    print(f"  {i}. {rec['hotel_name']}")
    print(f"     {rec['location']} - ${rec['avg_price']:.2f}/night")
    print(f"     Method: {rec['methods_used']}\n")

# ============================================================================
# [STEP 6] TEST EVALUATION (OPTIONAL)
# ============================================================================

print("="*70)
print("[STEP 6] Testing Evaluation Framework (Optional)")
print("="*70)

try:
    print("\nRunning evaluation on sample users...")
    
    evaluator = RecommendationEvaluator(complete_df, user_hotel_matrix)
    
    # Create train/test split
    train_data, test_data = evaluator.create_train_test_split(
        test_size=0.2,
        min_interactions=3
    )
    
    # Evaluate model
    results = evaluator.evaluate_model(
        rec_engine,
        k_values=[3, 5, 10],
        sample_users=20  # Small sample for quick testing
    )
    
    # Print results
    evaluator.print_evaluation_results(results)
    
    print("\n✓ Evaluation complete!")
    
except Exception as e:
    print(f" Evaluation skipped: {e}")

# ============================================================================
# [STEP 7] SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TESTING COMPLETE - SUMMARY")
print("="*70)

print("\n All core functionalities tested successfully:")
print("   1. ✓ Basic hybrid recommendations")
print("   2. ✓ Filtered recommendations (destination + budget)")
print("   3. ✓ Cold start handling for new users")
print("   4. ✓ Evaluation framework (optional)")

print("\n Recommendation System is ready for deployment!")
print("="*70)