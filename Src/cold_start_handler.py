import pandas as pd
import numpy as np

class ColdStartHandler:
    """
    Handles recommendations for new users and new hotels
    """
    
    def __init__(self, complete_df, hotel_features, users_df):
        self.complete_df = complete_df
        self.hotel_features = hotel_features
        self.users_df = users_df
        
        # Pre-compute demographic-based preferences
        self._precompute_demographic_preferences()
    
    def _precompute_demographic_preferences(self):
        """
        Pre-compute hotel preferences by gender and age group
        """
        # Gender-based preferences
        self.gender_preferences = self.complete_df.groupby(['gender', 'hotel_name']).agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        
        # FIXED: Rename columns properly
        self.gender_preferences.columns = ['gender', 'hotel_name', 'booking_count', 'avg_price', 'avg_days']
        
        # Age-based preferences
        self.complete_df['age_group'] = pd.cut(
            self.complete_df['age'],
            bins=[0, 25, 35, 50, float('inf')],
            labels=['Young', 'Adult', 'Middle-Aged', 'Senior']
        )

        self.age_preferences = self.complete_df.groupby(['age_group', 'hotel_name'], observed=False).agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        
        # FIXED: Rename columns properly
        self.age_preferences.columns = ['age_group', 'hotel_name', 'booking_count', 'avg_price', 'avg_days']
        
        # Combined gender-age preferences
        self.gender_age_preferences = self.complete_df.groupby(['gender', 'age_group', 'hotel_name'], observed=False).agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        
        # FIXED: Rename columns properly
        self.gender_age_preferences.columns = ['gender', 'age_group', 'hotel_name', 'booking_count', 'avg_price', 'avg_days']
    
    def get_new_user_recommendations(self, user_demographics, destination=None, 
                                     budget_min=0, budget_max=float('inf'), 
                                     n_recommendations=10):
        """
        Get recommendations for a completely new user based on demographics
        
        Args:
            user_demographics (dict): {'gender': str, 'age': int}
            destination (str): Optional destination filter
            budget_min (float): Minimum budget
            budget_max (float): Maximum budget
            n_recommendations (int): Number of recommendations
        
        Returns:
            list: Recommended hotels
        """
        gender = user_demographics.get('gender', '').lower()
        age = user_demographics.get('age', 30)
        
        # Determine age group
        if age < 25:
            age_group = 'Young'
        elif age < 35:
            age_group = 'Adult'
        elif age < 50:
            age_group = 'Middle-Aged'
        else:
            age_group = 'Senior'
        
        # Try gender + age group first
        gender_age_prefs = self.gender_age_preferences[
            (self.gender_age_preferences['gender'] == gender) &
            (self.gender_age_preferences['age_group'] == age_group)
        ].copy()
        
        # If not enough data, fallback to gender only
        if len(gender_age_prefs) < n_recommendations:
            gender_prefs = self.gender_preferences[
                self.gender_preferences['gender'] == gender
            ].copy()
            
            # Use gender preferences
            recommendations_df = gender_prefs
        else:
            recommendations_df = gender_age_prefs
        
        # If still no data, use overall popular hotels
        if len(recommendations_df) == 0:
            recommendations_df = self.hotel_features.copy()
            recommendations_df['booking_count'] = recommendations_df['booking_count']
            # avg_price and avg_days already exist in hotel_features
        
        # FIXED: Check if columns exist before filtering
        if 'avg_price' in recommendations_df.columns:
            # Apply budget filter
            recommendations_df = recommendations_df[
                (recommendations_df['avg_price'] >= budget_min) &
                (recommendations_df['avg_price'] <= budget_max)
            ]
        
        # Apply destination filter if provided
        if destination:
            # Merge with hotel_features to get location
            recommendations_df = recommendations_df.merge(
                self.hotel_features[['hotel_name', 'location']], 
                on='hotel_name', 
                how='left'
            )
            
            destination_clean = str(destination).strip()
            recommendations_df = recommendations_df[
                recommendations_df['location'].str.strip() == destination_clean
            ]
        
        # Sort by booking count (popularity)
        recommendations_df = recommendations_df.sort_values(
            'booking_count', 
            ascending=False
        ).head(n_recommendations)
        
        # FIXED: Merge with hotel_features to get all required columns
        final_recs = recommendations_df.merge(
            self.hotel_features[['hotel_name', 'location', 'avg_price', 'avg_stay', 'booking_count']], 
            on='hotel_name', 
            how='left',
            suffixes=('', '_hotel')
        )
        
        # Use hotel_features data if available
        if 'avg_price_hotel' in final_recs.columns:
            final_recs['avg_price'] = final_recs['avg_price_hotel'].fillna(final_recs['avg_price'])
            final_recs = final_recs.drop(columns=['avg_price_hotel'])
        
        if 'booking_count_hotel' in final_recs.columns:
            final_recs['booking_count'] = final_recs['booking_count_hotel'].fillna(final_recs['booking_count'])
            final_recs = final_recs.drop(columns=['booking_count_hotel'])
        
        # Create recommendation output
        recommendations = []
        for _, row in final_recs.iterrows():
            rec = {
                'hotel_name': row['hotel_name'],
                'location': row.get('location', 'Unknown'),
                'avg_price': round(float(row.get('avg_price', 0)), 2),
                'avg_stay': round(float(row.get('avg_stay', row.get('avg_days', 0))), 2),
                'popularity': int(row.get('booking_count', 0)),
                'recommendation_score': round(float(row.get('booking_count', 0)) / 
                                            self.hotel_features['booking_count'].max(), 4),
                'methods_used': f'cold-start (demographics: {gender}, {age_group})'
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_new_hotel_recommendations(self, hotel_info, similar_users=None, n_users=10):
        """
        Get potential users who might be interested in a new hotel
        
        Args:
            hotel_info (dict): {'location': str, 'price': float, 'category': str}
            similar_users (list): Optional list of user codes
            n_users (int): Number of users to recommend to
        
        Returns:
            list: User codes who might be interested
        """
        location = hotel_info.get('location')
        price = hotel_info.get('price', 0)
        
        # Find users who have booked in similar location or price range
        similar_bookings = self.complete_df[
            (self.complete_df['place'] == location) |
            (abs(self.complete_df['price'] - price) / max(price, 1) < 0.3)  # Within 30% of price
        ]
        
        # Get most frequent users
        user_counts = similar_bookings['userCode'].value_counts().head(n_users)
        
        return user_counts.index.tolist()
    
    def is_new_user(self, user_code):
        """
        Check if a user is new (not in the system)
        """
        return user_code not in self.users_df['code'].values
    
    def is_new_hotel(self, hotel_name):
        """
        Check if a hotel is new (not in the system)
        """
        return hotel_name not in self.hotel_features['hotel_name'].values


# ============================================================================
# ENHANCED RECOMMENDATION ENGINE WITH COLD START HANDLING
# ============================================================================

class EnhancedHotelRecommendationEngine:
    """
    Enhanced recommendation engine with cold start handling
    """
    
    def __init__(self, base_engine, cold_start_handler):
        self.base_engine = base_engine
        self.cold_start = cold_start_handler
    
    def get_recommendations(self, user_code=None, user_demographics=None,
                          destination=None, budget_min=0, budget_max=float('inf'),
                          n_recommendations=10, apply_diversity=True, debug=False):
        """
        Get recommendations with automatic cold start detection
        
        Args:
            user_code (str): Existing user code (optional)
            user_demographics (dict): For new users: {'gender': str, 'age': int}
            destination (str): Optional destination filter
            budget_min (float): Minimum budget
            budget_max (float): Maximum budget
            n_recommendations (int): Number of recommendations
            apply_diversity (bool): Apply diversity filter
            debug (bool): Print debug information
        
        Returns:
            list: Hotel recommendations
        """
        # Check if it's a new user
        if user_code and self.cold_start.is_new_user(user_code):
            if debug:
                print(f"   COLD START: New user detected (code: {user_code})")
                print(f"   Using demographic-based recommendations")
            
            # Get user demographics if not provided
            if not user_demographics:
                user_info = self.base_engine.get_user_info(user_code)
                if user_info:
                    user_demographics = {
                        'gender': user_info['gender'],
                        'age': user_info['age']
                    }
                else:
                    # Use default demographics
                    user_demographics = {'gender': 'unknown', 'age': 30}
            
            return self.cold_start.get_new_user_recommendations(
                user_demographics,
                destination=destination,
                budget_min=budget_min,
                budget_max=budget_max,
                n_recommendations=n_recommendations
            )
        
        # New user with demographics provided
        elif user_demographics and not user_code:
            if debug:
                print(f"   COLD START: New user detected")
                print(f"   Using demographic-based recommendations")
            
            return self.cold_start.get_new_user_recommendations(
                user_demographics,
                destination=destination,
                budget_min=budget_min,
                budget_max=budget_max,
                n_recommendations=n_recommendations
            )
        
        # Existing user - use base engine
        else:
            return self.base_engine.hybrid_recommendations(
                user_code,
                destination=destination,
                budget_min=budget_min,
                budget_max=budget_max,
                n_recommendations=n_recommendations,
                apply_diversity=apply_diversity,
                debug=debug
            )
    
    def get_recommendations_summary(self, user_code=None, user_demographics=None,
                                   n_recommendations=10):
        """
        Get comprehensive recommendation summary
        """
        if user_code and not self.cold_start.is_new_user(user_code):
            return self.base_engine.get_recommendations_summary(user_code, n_recommendations)
        else:
            # For new users, create a basic summary
            if not user_demographics:
                user_demographics = {'gender': 'unknown', 'age': 30}
            
            recommendations = self.get_recommendations(
                user_demographics=user_demographics,
                n_recommendations=n_recommendations
            )
            
            return {
                'user_info': {
                    'code': 'NEW_USER',
                    'gender': user_demographics.get('gender', 'unknown'),
                    'age': user_demographics.get('age', 'unknown')
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
# TESTING EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Testing Cold Start Handler...")
    
    # You would load your data here
    # Example usage:
    """
    # Load data
    models = load_recommendation_models()
    
    # Create cold start handler
    cold_start = ColdStartHandler(
        complete_df=models['complete_data'],
        hotel_features=models['hotel_features'],
        users_df=models['users_data']
    )
    
    # Create enhanced engine
    enhanced_engine = EnhancedHotelRecommendationEngine(
        base_engine=rec_engine,  # Your base recommendation engine
        cold_start_handler=cold_start
    )
    
    # Test with new user
    new_user_recs = enhanced_engine.get_recommendations(
        user_demographics={'gender': 'female', 'age': 28},
        budget_min=100,
        budget_max=300,
        n_recommendations=5,
        debug=True
    )
    
    for i, rec in enumerate(new_user_recs, 1):
        print(f"{i}. {rec['hotel_name']} - ${rec['avg_price']}/night")
    """
    
    print("Cold Start Handler class defined successfully!")