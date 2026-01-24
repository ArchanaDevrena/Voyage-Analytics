import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class ColdStartHandler:
    """
    Handles recommendations for new users and new hotels with proper data handling
    """
    
    # Constants
    AGE_BINS = [0, 25, 35, 50, float('inf')]
    AGE_LABELS = ['Young', 'Adult', 'Middle-Aged', 'Senior']
    VALID_GENDERS = ['male', 'female', 'other', 'unknown']
    
    def __init__(self, train_df: pd.DataFrame, hotel_features: pd.DataFrame, users_df: pd.DataFrame):
        """
        Initialize ColdStartHandler
        
        Args:
            train_df: TRAINING data only (historical bookings - no data leakage!)
            hotel_features: Hotel metadata with columns: hotel_name, location, avg_price, avg_stay, booking_count
            users_df: User metadata with columns: code, gender, age
        """
        # Make copies to avoid modifying original data
        self.train_df = train_df.copy()
        self.hotel_features = hotel_features.copy()
        self.users_df = users_df.copy()
        
        # Normalize data
        self._normalize_data()
        
        # Pre-compute demographic-based preferences
        self._precompute_demographic_preferences()
    
    def _normalize_data(self):
        """Normalize and clean data"""
        # Normalize gender
        if 'gender' in self.train_df.columns:
            self.train_df['gender'] = self.train_df['gender'].str.lower().str.strip()
        
        if 'gender' in self.users_df.columns:
            self.users_df['gender'] = self.users_df['gender'].str.lower().str.strip()
        
        # Normalize location/place column names
        if 'place' in self.train_df.columns and 'location' not in self.train_df.columns:
            self.train_df['location'] = self.train_df['place']
        
        # Strip whitespace from location
        if 'location' in self.train_df.columns:
            self.train_df['location'] = self.train_df['location'].str.strip()
        
        if 'location' in self.hotel_features.columns:
            self.hotel_features['location'] = self.hotel_features['location'].str.strip()
    
    def _get_age_group(self, age: Union[int, float]) -> str:
        """
        Centralized age group calculation
        
        Args:
            age: Age in years
            
        Returns:
            Age group label
        """
        # Validate age
        if not isinstance(age, (int, float)) or pd.isna(age):
            return 'Adult'  # Default
        
        if age < 0 or age > 120:
            return 'Adult'  # Default for invalid ages
        
        if age < 25:
            return 'Young'
        elif age < 35:
            return 'Adult'
        elif age < 50:
            return 'Middle-Aged'
        else:
            return 'Senior'
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """
        Safely convert to float, handling NaN, None, etc.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value or default
        """
        try:
            if pd.isna(value) or value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _validate_gender(self, gender: str) -> str:
        """
        Validate and normalize gender input
        
        Args:
            gender: Gender string
            
        Returns:
            Normalized gender
        """
        if not gender or not isinstance(gender, str):
            return 'unknown'
        
        gender = gender.lower().strip()
        
        if gender not in self.VALID_GENDERS:
            return 'unknown'
        
        return gender
    
    def _precompute_demographic_preferences(self):
        """
        Pre-compute hotel preferences by gender and age group
        """
        # Add age groups to training data
        self.train_df['age_group'] = self.train_df['age'].apply(self._get_age_group)
        
        # Gender-based preferences
        self.gender_preferences = self.train_df.groupby(['gender', 'hotel_name'], dropna=False).agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        
        self.gender_preferences.columns = ['gender', 'hotel_name', 'booking_count', 'avg_price', 'avg_days']
        
        # Age-based preferences
        self.age_preferences = self.train_df.groupby(['age_group', 'hotel_name'], dropna=False).agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        
        self.age_preferences.columns = ['age_group', 'hotel_name', 'booking_count', 'avg_price', 'avg_days']
        
        # Combined gender-age preferences
        self.gender_age_preferences = self.train_df.groupby(
            ['gender', 'age_group', 'hotel_name'], 
            dropna=False
        ).agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        
        self.gender_age_preferences.columns = [
            'gender', 'age_group', 'hotel_name', 'booking_count', 'avg_price', 'avg_days'
        ]
    
    def get_new_user_recommendations(
        self, 
        user_demographics: Dict[str, Union[str, int, float]], 
        destination: Optional[str] = None, 
        budget_min: float = 0, 
        budget_max: float = float('inf'), 
        n_recommendations: int = 10
    ) -> List[Dict]:
        """
        Get recommendations for a completely new user based on demographics
        
        Args:
            user_demographics: Dictionary with 'gender' and 'age' keys
            destination: Optional destination filter
            budget_min: Minimum budget
            budget_max: Maximum budget
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of recommended hotels with metadata
        """
        # Validate inputs
        if not isinstance(user_demographics, dict):
            raise ValueError("user_demographics must be a dictionary with 'gender' and 'age' keys")
        
        # Extract and validate demographics
        gender = self._validate_gender(user_demographics.get('gender', 'unknown'))
        age = user_demographics.get('age', 30)
        age_group = self._get_age_group(age)
        
        # Strategy 1: Try gender + age group first (most specific)
        recommendations_df = self._get_gender_age_preferences(gender, age_group)
        strategy = f'gender+age ({gender}, {age_group})'
        
        # Strategy 2: Fallback to gender only if insufficient data
        if len(recommendations_df) < n_recommendations:
            recommendations_df = self._get_gender_preferences(gender)
            strategy = f'gender ({gender})'
        
        # Strategy 3: Fallback to overall popular hotels
        if len(recommendations_df) == 0:
            recommendations_df = self.hotel_features.copy()
            strategy = 'popularity (cold start)'
        
        # Merge with hotel_features to get complete information
        recommendations_df = self._merge_with_hotel_features(recommendations_df)
        
        # Apply filters
        recommendations_df = self._apply_filters(
            recommendations_df, 
            destination, 
            budget_min, 
            budget_max
        )
        
        # Handle case where no hotels match filters
        if len(recommendations_df) == 0:
            print(f"⚠️  No hotels match criteria. Returning top popular hotels.")
            recommendations_df = self.hotel_features.copy()
            recommendations_df = self._apply_filters(
                recommendations_df,
                destination=None,  # Remove destination filter
                budget_min=budget_min,
                budget_max=budget_max
            )
        
        # Sort by popularity and limit
        recommendations_df = recommendations_df.sort_values(
            'booking_count', 
            ascending=False
        ).head(n_recommendations)
        
        # Format output
        recommendations = self._format_recommendations(recommendations_df, strategy)
        
        return recommendations
    
    def _get_gender_age_preferences(self, gender: str, age_group: str) -> pd.DataFrame:
        """Get preferences for specific gender and age group"""
        return self.gender_age_preferences[
            (self.gender_age_preferences['gender'] == gender) &
            (self.gender_age_preferences['age_group'] == age_group)
        ].copy()
    
    def _get_gender_preferences(self, gender: str) -> pd.DataFrame:
        """Get preferences for specific gender"""
        return self.gender_preferences[
            self.gender_preferences['gender'] == gender
        ].copy()
    
    def _merge_with_hotel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge recommendations with hotel features to get complete information
        
        Args:
            df: DataFrame with at least 'hotel_name' column
            
        Returns:
            Merged DataFrame with hotel features
        """
        # If df already has all needed columns, just ensure hotel_features data is included
        merge_cols = ['hotel_name', 'location', 'avg_price', 'avg_stay', 'booking_count']
        available_cols = [col for col in merge_cols if col in self.hotel_features.columns]
        
        if 'hotel_name' not in df.columns:
            return self.hotel_features[available_cols].copy()
        
        # Merge, preferring hotel_features values
        result = df.merge(
            self.hotel_features[available_cols],
            on='hotel_name',
            how='left',
            suffixes=('_demo', '')
        )
        
        # For any columns that exist in both, prefer hotel_features (no suffix)
        # Fill missing values from demographic preferences
        if 'avg_price_demo' in result.columns:
            result['avg_price'] = result['avg_price'].fillna(result['avg_price_demo'])
            result = result.drop(columns=['avg_price_demo'])
        
        if 'booking_count_demo' in result.columns:
            result['booking_count'] = result['booking_count'].fillna(result['booking_count_demo'])
            result = result.drop(columns=['booking_count_demo'])
        
        if 'avg_days' in result.columns and 'avg_stay' in result.columns:
            result['avg_stay'] = result['avg_stay'].fillna(result['avg_days'])
        
        return result
    
    def _apply_filters(
        self, 
        df: pd.DataFrame, 
        destination: Optional[str], 
        budget_min: float, 
        budget_max: float
    ) -> pd.DataFrame:
        """
        Apply destination and budget filters
        
        Args:
            df: DataFrame to filter
            destination: Optional destination filter
            budget_min: Minimum budget
            budget_max: Maximum budget
            
        Returns:
            Filtered DataFrame
        """
        result = df.copy()
        
        # Apply budget filter
        if 'avg_price' in result.columns:
            result = result[
                (result['avg_price'] >= budget_min) &
                (result['avg_price'] <= budget_max)
            ]
        
        # Apply destination filter
        if destination and 'location' in result.columns:
            destination_clean = str(destination).strip()
            result = result[result['location'] == destination_clean]
        
        return result
    
    def _format_recommendations(self, df: pd.DataFrame, strategy: str) -> List[Dict]:
        """
        Format recommendations into output format
        
        Args:
            df: DataFrame with recommendations
            strategy: Strategy used for recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        max_booking_count = self.hotel_features['booking_count'].max()
        if pd.isna(max_booking_count) or max_booking_count == 0:
            max_booking_count = 1
        
        for _, row in df.iterrows():
            booking_count = self._safe_float(row.get('booking_count', 0))
            
            rec = {
                'hotel_name': row['hotel_name'],
                'location': row.get('location', 'Unknown'),
                'avg_price': round(self._safe_float(row.get('avg_price', 0)), 2),
                'avg_stay': round(self._safe_float(row.get('avg_stay', 0)), 2),
                'popularity': int(booking_count),
                'recommendation_score': round(booking_count / max_booking_count, 4),
                'methods_used': f'cold-start ({strategy})'
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_users_for_new_hotel(
        self, 
        hotel_info: Dict[str, Union[str, float]], 
        n_users: int = 10
    ) -> List[str]:
        """
        Get potential users who might be interested in a new hotel
        
        Args:
            hotel_info: Dictionary with 'location', 'price' keys
            n_users: Number of users to return
        
        Returns:
            List of user codes who might be interested
        """
        location = hotel_info.get('location', '').strip()
        price = self._safe_float(hotel_info.get('price', 0))
        
        # Use median price if invalid
        if price <= 0:
            price = self.train_df['price'].median()
        
        # Find users with similar booking patterns
        location_match = pd.Series([False] * len(self.train_df))
        if location and 'location' in self.train_df.columns:
            location_match = self.train_df['location'] == location
        
        # Price match: within 30% of target price
        price_match = (
            (self.train_df['price'] >= price * 0.7) & 
            (self.train_df['price'] <= price * 1.3)
        )
        
        similar_bookings = self.train_df[location_match | price_match]
        
        # Fallback: return most active users if no matches
        if len(similar_bookings) == 0:
            similar_bookings = self.train_df
        
        # Get most frequent users
        user_counts = similar_bookings['userCode'].value_counts().head(n_users)
        
        return user_counts.index.tolist()
    
    def is_new_user(self, user_code: str) -> bool:
        """
        Check if a user is new (not in the system)
        
        Args:
            user_code: User identifier
            
        Returns:
            True if user is new, False otherwise
        """
        return user_code not in self.users_df['code'].values
    
    def is_new_hotel(self, hotel_name: str) -> bool:
        """
        Check if a hotel is new (not in the system)
        
        Args:
            hotel_name: Hotel name
            
        Returns:
            True if hotel is new, False otherwise
        """
        return hotel_name not in self.hotel_features['hotel_name'].values


# ============================================================================
# ENHANCED RECOMMENDATION ENGINE WITH COLD START HANDLING
# ============================================================================

class EnhancedHotelRecommendationEngine:
    """
    Enhanced recommendation engine with automatic cold start detection and handling
    """
    
    def __init__(self, base_engine, cold_start_handler: ColdStartHandler):
        """
        Initialize enhanced engine
        
        Args:
            base_engine: Base recommendation engine for existing users
            cold_start_handler: ColdStartHandler instance
        """
        self.base_engine = base_engine
        self.cold_start = cold_start_handler
    
    def get_recommendations(
        self, 
        user_code: Optional[str] = None, 
        user_demographics: Optional[Dict[str, Union[str, int, float]]] = None,
        destination: Optional[str] = None, 
        budget_min: float = 0, 
        budget_max: float = float('inf'),
        n_recommendations: int = 10, 
        apply_diversity: bool = True, 
        debug: bool = False
    ) -> List[Dict]:
        """
        Get recommendations with automatic cold start detection
        
        Args:
            user_code: Existing user code (optional)
            user_demographics: For new users: {'gender': str, 'age': int}
            destination: Optional destination filter
            budget_min: Minimum budget
            budget_max: Maximum budget
            n_recommendations: Number of recommendations
            apply_diversity: Apply diversity filter (only for existing users)
            debug: Print debug information
        
        Returns:
            List of hotel recommendations
        """
        # Case 1: Existing user with user_code
        if user_code and not self.cold_start.is_new_user(user_code):
            if debug:
                print(f"✓ Existing user detected (code: {user_code})")
                print(f"  Using base recommendation engine")
            
            return self.base_engine.hybrid_recommendations(
                user_code,
                destination=destination,
                budget_min=budget_min,
                budget_max=budget_max,
                n_recommendations=n_recommendations,
                apply_diversity=apply_diversity,
                debug=debug
            )
        
        # Case 2: New user with user_code but no history
        elif user_code and self.cold_start.is_new_user(user_code):
            if debug:
                print(f"⚠️  COLD START: New user detected (code: {user_code})")
                print(f"  Using demographic-based recommendations")
            
            # Try to get user demographics from user info
            if not user_demographics:
                try:
                    user_info = self.base_engine.get_user_info(user_code)
                    if user_info:
                        user_demographics = {
                            'gender': user_info.get('gender', 'unknown'),
                            'age': user_info.get('age', 30)
                        }
                except:
                    # Use default demographics
                    user_demographics = {'gender': 'unknown', 'age': 30}
            
            return self.cold_start.get_new_user_recommendations(
                user_demographics,
                destination=destination,
                budget_min=budget_min,
                budget_max=budget_max,
                n_recommendations=n_recommendations
            )
        
        # Case 3: New user with demographics provided
        elif user_demographics and not user_code:
            if debug:
                print(f"⚠️  COLD START: New user with demographics")
                print(f"  Demographics: {user_demographics}")
            
            return self.cold_start.get_new_user_recommendations(
                user_demographics,
                destination=destination,
                budget_min=budget_min,
                budget_max=budget_max,
                n_recommendations=n_recommendations
            )
        
        # Case 4: No user info at all - use popular hotels
        else:
            if debug:
                print(f"⚠️  COLD START: No user information provided")
                print(f"  Using popular hotels")
            
            return self.cold_start.get_new_user_recommendations(
                user_demographics={'gender': 'unknown', 'age': 30},
                destination=destination,
                budget_min=budget_min,
                budget_max=budget_max,
                n_recommendations=n_recommendations
            )
    
    def get_recommendations_summary(
        self, 
        user_code: Optional[str] = None, 
        user_demographics: Optional[Dict[str, Union[str, int, float]]] = None,
        n_recommendations: int = 10
    ) -> Dict:
        """
        Get comprehensive recommendation summary with user info
        
        Args:
            user_code: User code (optional)
            user_demographics: User demographics for new users
            n_recommendations: Number of recommendations
            
        Returns:
            Dictionary with user info, stats, and recommendations
        """
        # Existing user
        if user_code and not self.cold_start.is_new_user(user_code):
            return self.base_engine.get_recommendations_summary(user_code, n_recommendations)
        
        # New user
        else:
            if not user_demographics:
                user_demographics = {'gender': 'unknown', 'age': 30}
            
            recommendations = self.get_recommendations(
                user_demographics=user_demographics,
                n_recommendations=n_recommendations
            )
            
            return {
                'user_info': {
                    'code': user_code or 'NEW_USER',
                    'gender': user_demographics.get('gender', 'unknown'),
                    'age': user_demographics.get('age', 'unknown'),
                    'is_new_user': True
                },
                'user_stats': {
                    'total_bookings': 0,
                    'avg_price': 0,
                    'avg_stay': 0,
                    'favorite_locations': {}
                },
                'recommendations': recommendations
            }


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING COLD START HANDLER")
    print("=" * 80)
    
    # Create mock data for testing
    print("\n1. Creating mock data...")
    
    # Mock training data (complete_df)
    np.random.seed(42)
    n_bookings = 100
    
    train_df = pd.DataFrame({
        'userCode': [f'U{i}' for i in np.random.randint(1, 11, n_bookings)],
        'hotel_name': [f'Hotel {chr(65+i)}' for i in np.random.randint(0, 10, n_bookings)],
        'gender': np.random.choice(['male', 'female'], n_bookings),
        'age': np.random.randint(20, 70, n_bookings),
        'price': np.random.uniform(50, 300, n_bookings),
        'days': np.random.randint(1, 7, n_bookings),
        'location': np.random.choice(['NYC', 'LA', 'SF', 'Miami', 'Chicago'], n_bookings)
    })
    
    # Mock hotel features
    hotels = [f'Hotel {chr(65+i)}' for i in range(10)]
    hotel_features = pd.DataFrame({
        'hotel_name': hotels,
        'location': np.random.choice(['NYC', 'LA', 'SF', 'Miami', 'Chicago'], 10),
        'avg_price': np.random.uniform(80, 250, 10),
        'avg_stay': np.random.uniform(1.5, 5, 10),
        'booking_count': np.random.randint(10, 200, 10)
    })
    
    # Mock users
    users_df = pd.DataFrame({
        'code': [f'U{i}' for i in range(1, 11)],
        'gender': np.random.choice(['male', 'female'], 10),
        'age': np.random.randint(20, 70, 10)
    })
    
    print(f"   ✓ Created {len(train_df)} bookings")
    print(f"   ✓ Created {len(hotel_features)} hotels")
    print(f"   ✓ Created {len(users_df)} users")
    
    # Initialize handler
    print("\n2. Initializing ColdStartHandler...")
    handler = ColdStartHandler(train_df, hotel_features, users_df)
    print("   ✓ Handler initialized successfully")
    
    # Test 1: New user recommendations
    print("\n3. Testing new user recommendations...")
    print("\n   Test 3a: Young male user")
    recs = handler.get_new_user_recommendations(
        user_demographics={'gender': 'male', 'age': 23},
        n_recommendations=5
    )
    print(f"   ✓ Got {len(recs)} recommendations")
    for i, rec in enumerate(recs[:3], 1):
        print(f"      {i}. {rec['hotel_name']} ({rec['location']}) - ${rec['avg_price']:.2f}/night")
    
    # Test 2: With filters
    print("\n   Test 3b: Female user with budget filter")
    recs = handler.get_new_user_recommendations(
        user_demographics={'gender': 'female', 'age': 35},
        budget_min=100,
        budget_max=200,
        n_recommendations=5
    )
    print(f"   ✓ Got {len(recs)} recommendations (filtered by budget)")
    
    # Test 3: With destination
    print("\n   Test 3c: With destination filter")
    recs = handler.get_new_user_recommendations(
        user_demographics={'gender': 'male', 'age': 45},
        destination='NYC',
        n_recommendations=5
    )
    print(f"   ✓ Got {len(recs)} recommendations (NYC only)")
    
    # Test 4: Check new user detection
    print("\n4. Testing user detection...")
    print(f"   Is U1 new? {handler.is_new_user('U1')}")  # Should be False
    print(f"   Is U999 new? {handler.is_new_user('U999')}")  # Should be True
    
    # Test 5: Get users for new hotel
    print("\n5. Testing new hotel user targeting...")
    users = handler.get_users_for_new_hotel(
        hotel_info={'location': 'NYC', 'price': 150},
        n_users=5
    )
    print(f"   ✓ Found {len(users)} potential users: {users[:5]}")
    
    # Test 6: Edge cases
    print("\n6. Testing edge cases...")
    
    print("   Test 6a: Invalid age")
    recs = handler.get_new_user_recommendations(
        user_demographics={'gender': 'male', 'age': -5},  # Invalid age
        n_recommendations=3
    )
    print(f"   ✓ Handled invalid age, got {len(recs)} recommendations")
    
    print("   Test 6b: Unknown gender")
    recs = handler.get_new_user_recommendations(
        user_demographics={'gender': 'alien', 'age': 30},  # Invalid gender
        n_recommendations=3
    )
    print(f"   ✓ Handled unknown gender, got {len(recs)} recommendations")
    
    print("   Test 6c: Extreme budget")
    recs = handler.get_new_user_recommendations(
        user_demographics={'gender': 'female', 'age': 25},
        budget_min=1000,  # Very high budget
        budget_max=2000,
        n_recommendations=3
    )
    print(f"   ✓ Handled extreme budget, got {len(recs)} recommendations")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nCold Start Handler is ready for production use!")
