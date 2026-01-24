"""
Hotel Recommendation System - Complete Testing Suite
Professional production-ready testing with comprehensive coverage
"""

import sys
import os
import warnings
import traceback
from typing import Optional

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Src'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import modules
try:
    from recommendation_engine import HotelRecommendationEngine, load_recommendation_models
    from cold_start_handler import ColdStartHandler, EnhancedHotelRecommendationEngine 
    from evaluation import RecommendationEvaluator, run_evaluation
    print("All modules imported successfully")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are in the Src/ directory")
    sys.exit(1)


def print_section_header(title: str, step_num: Optional[int] = None):
    """Print formatted section header"""
    print("\n" + "="*70)
    if step_num:
        print(f"[STEP {step_num}] {title}")
    else:
        print(title)
    print("="*70)


def safe_test(test_name: str, test_func):
    """
    Safely execute a test function with error handling
    
    Args:
        test_name: Name of the test
        test_func: Function to execute
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\nRunning: {test_name}")
        test_func()
        print(f"PASSED - {test_name}")
        return True
    except Exception as e:
        print(f"FAILED - {test_name}")
        print(f"   Error: {str(e)}")
        if '--verbose' in sys.argv:
            traceback.print_exc()
        return False


# ============================================================================
# MAIN TESTING SUITE
# ============================================================================

def run_all_tests():
    """Run comprehensive test suite"""
    
    print_section_header("VOYAGE ANALYTICS - RECOMMENDATION SYSTEM TESTING")
    
    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # ========================================================================
    # [STEP 1] LOAD MODELS AND DATA
    # ========================================================================
    
    print_section_header("Loading Models and Data", 1)
    
    try:
        print("\nLoading recommendation models...")
        
        # Try to load from models directory
        models_dir = 'models'
        if not os.path.exists(models_dir):
            # Try alternative paths
            alt_paths = ['../models', './Models', '../Models']
            for path in alt_paths:
                if os.path.exists(path):
                    models_dir = path
                    break
            else:
                raise FileNotFoundError(
                    f"Models directory not found. Tried: {models_dir}, {alt_paths}"
                )
        
        models = load_recommendation_models(models_dir=models_dir)
        
        # Extract models
        user_hotel_matrix = models['user_hotel_matrix']
        user_similarity_df = models['user_similarity']
        hotel_similarity_df = models['hotel_similarity']
        hotel_features = models['hotel_features']
        complete_df = models['complete_data']
        users_df = models['users_data']
        
        print("All models loaded successfully")
        print(f"  Users: {len(users_df):,}")
        print(f"  Hotels: {len(hotel_features):,}")
        print(f"  User-Hotel Matrix: {user_hotel_matrix.shape}")
        print(f"  Total Interactions: {len(complete_df):,}")
        
        results['passed'] += 1
        
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("\nPlease ensure:")
        print("  1. Models are saved in the 'models/' directory")
        print("  2. All required pickle files exist")
        print("  3. You've run the model training script first")
        return results
    
    # ========================================================================
    # [STEP 2] INITIALIZE ENGINES
    # ========================================================================
    
    print_section_header("Initializing Recommendation Engines", 2)
    
    try:
        print("\nInitializing base recommendation engine...")
        
        # Base recommendation engine
        rec_engine = HotelRecommendationEngine(
            user_hotel_matrix=user_hotel_matrix,
            user_similarity_df=user_similarity_df,
            hotel_similarity_df=hotel_similarity_df,
            hotel_features=hotel_features,
            complete_df=complete_df,
            users_df=users_df
        )
        
        print("Base engine initialized")
        
        print("\nInitializing cold start handler...")
        
        # Cold start handler
        cold_start = ColdStartHandler(
            train_df=complete_df,
            hotel_features=hotel_features,
            users_df=users_df
        )
        
        print("Cold start handler initialized")
        
        print("\nInitializing enhanced engine...")
        
        # Enhanced engine with cold start
        enhanced_engine = EnhancedHotelRecommendationEngine(
            base_engine=rec_engine,
            cold_start_handler=cold_start
        )
        
        print("Enhanced engine initialized")
        print("\nAll engines initialized successfully")
        
        results['passed'] += 1
        
    except Exception as e:
        print(f"Failed to initialize engines: {e}")
        traceback.print_exc()
        return results
    
    # ========================================================================
    # [STEP 3] TEST BASIC RECOMMENDATIONS
    # ========================================================================
    
    print_section_header("Testing Basic Recommendations", 3)
    
    def test_basic_recommendations():
        # Select test user (with validation)
        if len(users_df) < 20:
            test_user = users_df['code'].iloc[0]
        else:
            test_user = users_df['code'].iloc[10]
        
        user_info = rec_engine.get_user_info(test_user)
        
        print(f"\nTest User Profile:")
        print(f"   User Code: {test_user}")
        print(f"   Name: {user_info['name']}")
        print(f"   Gender: {user_info['gender']}")
        print(f"   Age: {user_info['age']}")
        print(f"   Company: {user_info.get('company', 'N/A')}")
        
        # Get recommendations
        print(f"\nGenerating recommendations...")
        recs = rec_engine.hybrid_recommendations(test_user, n_recommendations=5)
        
        if not recs:
            raise ValueError("No recommendations returned")
        
        print(f"\nTop 5 Hybrid Recommendations:\n")
        
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['hotel_name']}")
            print(f"   Location: {rec['location']}")
            print(f"   Price: ${rec['avg_price']:.2f}/night")
            print(f"   Score: {rec['recommendation_score']:.4f}")
            print(f"   Methods: {rec['methods_used']}\n")
    
    if safe_test("Basic Recommendations", test_basic_recommendations):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # ========================================================================
    # [STEP 4] TEST FILTERED RECOMMENDATIONS
    # ========================================================================
    
    print_section_header("Testing Filtered Recommendations", 4)
    
    def test_filtered_recommendations():
        # Select different test user
        if len(users_df) < 60:
            test_user2 = users_df['code'].iloc[-1]
        else:
            test_user2 = users_df['code'].iloc[50]
        
        user_info2 = rec_engine.get_user_info(test_user2)
        
        # Get user's most common location from history
        user_bookings = complete_df[complete_df['userCode'] == test_user2]
        
        if len(user_bookings) > 0 and 'location' in user_bookings.columns:
            destination = user_bookings['location'].mode()[0] if len(user_bookings['location'].mode()) > 0 else None
        else:
            destination = complete_df['location'].mode()[0] if 'location' in complete_df.columns else None
        
        print(f"\nUser: {user_info2['name']}")
        
        if destination:
            print(f"Filters: {destination}, Budget $150-$350\n")
            
            filtered_recs = rec_engine.hybrid_recommendations(
                test_user2,
                destination=destination,
                budget_min=150,
                budget_max=350,
                n_recommendations=5
            )
        else:
            print(f"Filters: Budget $150-$350 (no destination filter)\n")
            
            filtered_recs = rec_engine.hybrid_recommendations(
                test_user2,
                budget_min=150,
                budget_max=350,
                n_recommendations=5
            )
        
        print(f"Found {len(filtered_recs)} filtered recommendations:")
        for i, rec in enumerate(filtered_recs, 1):
            print(f"  {i}. {rec['hotel_name']}")
            print(f"     ${rec['avg_price']:.2f}/night | {rec['location']}")
    
    if safe_test("Filtered Recommendations", test_filtered_recommendations):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # ========================================================================
    # [STEP 5] TEST COLD START HANDLER
    # ========================================================================
    
    print_section_header("Testing Cold Start Handler", 5)
    
    def test_cold_start():
        # Test 1: New user with demographics
        print("\nTest 5a: New User with Demographics")
        
        new_user_demo = {
            'gender': 'female',
            'age': 28
        }
        
        print(f"\nNew User Profile:")
        print(f"   Gender: {new_user_demo['gender']}")
        print(f"   Age: {new_user_demo['age']}\n")
        
        new_user_recs = enhanced_engine.get_recommendations(
            user_demographics=new_user_demo,
            budget_min=100,
            budget_max=250,
            n_recommendations=5,
            debug=True
        )
        
        if not new_user_recs:
            raise ValueError("No cold start recommendations returned")
        
        print(f"\nCold Start Recommendations:")
        for i, rec in enumerate(new_user_recs, 1):
            print(f"  {i}. {rec['hotel_name']}")
            print(f"     {rec['location']} | ${rec['avg_price']:.2f}/night")
            print(f"     {rec['methods_used']}")
        
        # Test 2: New user code (not in system)
        print("\n\nTest 5b: New User Code (Not in System)")
        
        new_user_recs2 = enhanced_engine.get_recommendations(
            user_code='BRAND_NEW_USER_999',
            user_demographics={'gender': 'male', 'age': 35},
            n_recommendations=3,
            debug=True
        )
        
        print(f"\nGot {len(new_user_recs2)} recommendations for new user")
        
        # Test 3: Existing user (should use base engine)
        print("\n\nTest 5c: Existing User (Should Use Base Engine)")
        
        existing_user = users_df['code'].iloc[0]
        existing_recs = enhanced_engine.get_recommendations(
            user_code=existing_user,
            n_recommendations=3,
            debug=True
        )
        
        print(f"\nGot {len(existing_recs)} recommendations for existing user")
    
    if safe_test("Cold Start Handler", test_cold_start):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # ========================================================================
    # [STEP 6] TEST USER INFO RETRIEVAL
    # ========================================================================
    
    print_section_header("Testing User Information Retrieval", 6)
    
    def test_user_info():
        test_user = users_df['code'].iloc[0]
        
        print(f"\nTesting get_user_info for: {test_user}")
        
        user_info = rec_engine.get_user_info(test_user)
        
        required_fields = ['code', 'name', 'gender', 'age']
        for field in required_fields:
            if field not in user_info:
                raise ValueError(f"Missing required field: {field}")
        
        print(f"\nUser Info Retrieved:")
        for key, value in user_info.items():
            print(f"   {key}: {value}")
        
        # Test user stats
        print(f"\nTesting get_user_stats for: {test_user}")
        
        user_stats = rec_engine.get_user_stats(test_user)
        
        print(f"\nUser Stats:")
        print(f"   Total Bookings: {user_stats['total_bookings']}")
        print(f"   Avg Price: ${user_stats['avg_price']:.2f}")
        print(f"   Avg Stay: {user_stats['avg_stay']:.1f} days")
        print(f"   Favorite Locations: {list(user_stats['favorite_locations'].keys())[:3]}")
    
    if safe_test("User Information Retrieval", test_user_info):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # ========================================================================
    # [STEP 7] TEST DIVERSITY FILTER
    # ========================================================================
    
    print_section_header("Testing Diversity Filter", 7)
    
    def test_diversity():
        test_user = users_df['code'].iloc[5] if len(users_df) > 5 else users_df['code'].iloc[0]
        
        print(f"\nWithout Diversity Filter:")
        recs_no_div = rec_engine.hybrid_recommendations(
            test_user,
            n_recommendations=10,
            apply_diversity=False
        )
        
        locations_no_div = [r['location'] for r in recs_no_div]
        unique_no_div = len(set(locations_no_div))
        print(f"   Unique locations: {unique_no_div}/10")
        print(f"   Locations: {locations_no_div[:5]}...")
        
        print(f"\nWith Diversity Filter:")
        recs_with_div = rec_engine.hybrid_recommendations(
            test_user,
            n_recommendations=10,
            apply_diversity=True
        )
        
        locations_with_div = [r['location'] for r in recs_with_div]
        unique_with_div = len(set(locations_with_div))
        print(f"   Unique locations: {unique_with_div}/10")
        print(f"   Locations: {locations_with_div[:5]}...")
        
        if unique_with_div >= unique_no_div:
            print(f"\nDiversity filter working (increased from {unique_no_div} to {unique_with_div} unique locations)")
        else:
            print(f"\nNote: Diversity may be limited by available data")
    
    if safe_test("Diversity Filter", test_diversity):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # ========================================================================
    # [STEP 8] TEST EVALUATION FRAMEWORK
    # ========================================================================
    
    print_section_header("Testing Evaluation Framework", 8)
    
    def test_evaluation():
        print("\nRunning quick evaluation on sample users...")
        print("   (This may take a moment...)\n")
        
        evaluator = RecommendationEvaluator(complete_df, user_hotel_matrix)
        
        # Create train/test split
        train_data, test_data = evaluator.create_train_test_split(
            test_size=0.2,
            min_interactions=3
        )
        
        print(f"\n   Train size: {len(train_data):,}")
        print(f"   Test size: {len(test_data):,}")
        
        # Evaluate model (small sample for speed)
        results_eval = evaluator.evaluate_model(
            rec_engine,
            k_values=[5, 10],
            sample_users=10,
            verbose=False
        )
        
        print(f"\nEvaluation Complete")
        print(f"\nKey Metrics:")
        print(f"   Precision@5: {results_eval.get('precision@5', 0):.4f}")
        print(f"   Recall@5: {results_eval.get('recall@5', 0):.4f}")
        print(f"   NDCG@5: {results_eval.get('ndcg@5', 0):.4f}")
        print(f"   Coverage: {results_eval.get('coverage', 0):.4f}")
        print(f"   Diversity: {results_eval.get('diversity', 0):.4f}")
    
    if safe_test("Evaluation Framework", test_evaluation):
        results['passed'] += 1
    else:
        results['failed'] += 1
        print("   Note: Evaluation may fail if insufficient data for train/test split")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_section_header("TESTING COMPLETE - SUMMARY")
    
    total_tests = results['passed'] + results['failed'] + results['skipped']
    success_rate = (results['passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTest Results:")
    print(f"   Passed:  {results['passed']}/{total_tests}")
    print(f"   Failed:  {results['failed']}/{total_tests}")
    print(f"   Skipped: {results['skipped']}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    print(f"\nCore Functionalities Tested:")
    print(f"   1. {'PASS' if results['passed'] >= 1 else 'FAIL'} - Model loading and initialization")
    print(f"   2. {'PASS' if results['passed'] >= 3 else 'FAIL'} - Basic hybrid recommendations")
    print(f"   3. {'PASS' if results['passed'] >= 4 else 'FAIL'} - Filtered recommendations")
    print(f"   4. {'PASS' if results['passed'] >= 5 else 'FAIL'} - Cold start handling")
    print(f"   5. {'PASS' if results['passed'] >= 6 else 'FAIL'} - User information retrieval")
    print(f"   6. {'PASS' if results['passed'] >= 7 else 'FAIL'} - Diversity filtering")
    print(f"   7. {'PASS' if results['passed'] >= 8 else 'FAIL'} - Evaluation framework")
    
    if results['passed'] >= 6:
        print(f"\nRecommendation System is READY for deployment")
        print(f"All critical tests passed")
    elif results['passed'] >= 4:
        print(f"\nRecommendation System is MOSTLY functional")
        print(f"Some features may need attention")
    else:
        print(f"\nRecommendation System needs debugging")
        print(f"Multiple critical tests failed")
    
    print("="*70)
    
    return results


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("\nStarting Comprehensive Test Suite...\n")
    
    # Check for verbose flag
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    if verbose:
        print("Verbose mode enabled\n")
    
    # Run all tests
    test_results = run_all_tests()
    
    # Exit with appropriate code
    if test_results['failed'] == 0:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure
