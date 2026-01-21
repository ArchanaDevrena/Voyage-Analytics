import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging

# ============================================================================
# SETUP LOGGING (MODULE LEVEL - OUTSIDE CLASS)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HotelRecommendationEngine:
    """
    Optimized Hybrid Hotel Recommendation System
    Combines: Collaborative Filtering + Content-Based + Gender-Age-Based preferences
    
    Tuned for small catalogs with sparse interaction data.
    Focus: Simplicity, reliability, and balanced diversity.
    """
    
    def __init__(self, user_hotel_matrix, user_similarity_df, 
                 hotel_similarity_df, hotel_features, complete_df, users_df):
        self.user_hotel_matrix = user_hotel_matrix
        self.user_similarity_df = user_similarity_df
        self.hotel_similarity_df = hotel_similarity_df
        
        # Normalize hotel_features locations on initialization
        self.hotel_features = hotel_features.copy()
        self.hotel_features['location'] = self.hotel_features['location'].astype(str).str.strip()
        
        self.complete_df = complete_df
        self.users_df = users_df
        logger.info("HotelRecommendationEngine initialized successfully")
        
    def get_user_info(self, user_code):
        """Get user demographic information"""
        user_data = self.users_df[self.users_df['code'] == user_code]
        if len(user_data) == 0:
            logger.warning(f"User not found: {user_code}")
            return None
        return {
            'code': user_code,
            'name': user_data['name'].values[0],
            'gender': user_data['gender'].values[0],
            'age': user_data['age'].values[0],
            'company': user_data['company'].values[0]
        }
    
    def get_user_booking_count(self, user_code):
        """Get number of bookings for experience-based adjustments"""
        user_bookings = self.complete_df[self.complete_df['userCode'] == user_code]
        return len(user_bookings)
    
    def validate_inputs(self, user_code=None, destination=None, budget_min=0, budget_max=float('inf')):
        """Validate all inputs before processing"""
        
        if user_code and not isinstance(user_code, str):
            raise ValueError("user_code must be a string")
        
        if budget_min < 0:
            raise ValueError("budget_min cannot be negative")
        
        if budget_max < budget_min:
            raise ValueError("budget_max must be >= budget_min")
        
        if budget_max > 10000:
            raise ValueError("budget_max seems unreasonably high")
        
        if destination:
            destination_normalized = str(destination).strip()
            valid_locations = self.hotel_features['location'].unique()
            if destination_normalized not in valid_locations:
                raise ValueError(f"Invalid destination. Must be one of: {list(valid_locations)}")
        
        logger.debug(f"Input validation passed - user: {user_code}, destination: {destination}, budget: ${budget_min}-${budget_max}")
        return True
    
    def collaborative_filtering_recommendations(self, user_code, n_recommendations=10):
        """
        Collaborative Filtering: Recommend hotels based on similar users' preferences
        """
        if user_code not in self.user_similarity_df.index:
            logger.warning(f"User {user_code} not in similarity matrix")
            return []
        
        # Get similar users (top 10)
        similar_users = self.user_similarity_df[user_code].sort_values(ascending=False)[1:11]
        
        # Get hotels liked by similar users but not yet tried by target user
        user_hotels = set(self.user_hotel_matrix.loc[user_code][self.user_hotel_matrix.loc[user_code] > 0].index)
        
        recommendations = {}
        for similar_user_code, similarity_score in similar_users.items():
            if similar_user_code in self.user_hotel_matrix.index:
                similar_user_hotels = self.user_hotel_matrix.loc[similar_user_code]
                for hotel, rating in similar_user_hotels.items():
                    if rating > 0 and hotel not in user_hotels:
                        if hotel not in recommendations:
                            recommendations[hotel] = 0
                        recommendations[hotel] += rating * similarity_score
        
        # Sort and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        logger.info(f"Collaborative filtering: {len(sorted_recs)} recommendations for user {user_code}")
        return [{'hotel_name': hotel, 'score': score, 'method': 'collaborative'} 
                for hotel, score in sorted_recs]
    
    def content_based_recommendations(self, user_code, n_recommendations=10):
        """
        Content-Based Filtering: Recommend similar hotels to those user has liked
        """
        if user_code not in self.user_hotel_matrix.index:
            return []
        
        # Get user's previously liked hotels
        user_hotels = self.user_hotel_matrix.loc[user_code]
        liked_hotels = user_hotels[user_hotels > 0].index.tolist()
        
        if not liked_hotels:
            logger.info(f"No liked hotels found for user {user_code}")
            return []
        
        # Find similar hotels
        recommendations = {}
        for hotel in liked_hotels:
            if hotel in self.hotel_similarity_df.index:
                similar_hotels = self.hotel_similarity_df[hotel].sort_values(ascending=False)[1:6]
                for similar_hotel, similarity_score in similar_hotels.items():
                    if similar_hotel not in liked_hotels:
                        if similar_hotel not in recommendations:
                            recommendations[similar_hotel] = 0
                        recommendations[similar_hotel] += similarity_score * user_hotels[hotel]
        
        # Sort and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        logger.info(f"Content-based filtering: {len(sorted_recs)} recommendations for user {user_code}")
        return [{'hotel_name': hotel, 'score': score, 'method': 'content-based'} 
                for hotel, score in sorted_recs]
    
    def gender_based_recommendations(self, user_code, n_recommendations=10):
        """
        Gender-Based Filtering with Age Grouping
        Recommend hotels popular among same gender AND similar age group
        """
        user_info = self.get_user_info(user_code)
        if not user_info:
            return []
        
        user_gender = user_info['gender']
        user_age = user_info['age']
        
        # Define age groups
        if user_age < 30:
            age_group = 'young'
            age_range = (18, 35)
        elif user_age < 45:
            age_group = 'adult'
            age_range = (30, 50)
        elif user_age < 60:
            age_group = 'middle'
            age_range = (45, 65)
        else:
            age_group = 'senior'
            age_range = (55, 100)
        
        # Get hotels popular among same gender AND age group
        gender_age_df = self.complete_df[
            (self.complete_df['gender'] == user_gender) &
            (self.complete_df['age'] >= age_range[0]) &
            (self.complete_df['age'] <= age_range[1])
        ]
        
        if len(gender_age_df) == 0:
            # Fallback to gender only
            gender_age_df = self.complete_df[self.complete_df['gender'] == user_gender]
        
        hotel_popularity = gender_age_df.groupby('hotel_name').agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        hotel_popularity.columns = ['hotel_name', 'popularity', 'avg_price', 'avg_days']
        
        # Filter out hotels user has already visited
        if user_code in self.user_hotel_matrix.index:
            user_hotels = set(self.user_hotel_matrix.loc[user_code][self.user_hotel_matrix.loc[user_code] > 0].index)
            hotel_popularity = hotel_popularity[~hotel_popularity['hotel_name'].isin(user_hotels)]
        
        # Sort by popularity
        hotel_popularity = hotel_popularity.sort_values('popularity', ascending=False).head(n_recommendations)
        
        logger.info(f"Gender-based filtering: {len(hotel_popularity)} recommendations for {user_gender}, age {user_age}")
        return [{'hotel_name': row['hotel_name'], 
                 'score': row['popularity'], 
                 'method': 'gender-based',
                 'avg_price': row['avg_price'],
                 'age_group': age_group} 
                for _, row in hotel_popularity.iterrows()]
    
    def location_based_recommendations(self, destination, budget_min=0, budget_max=float('inf'), n_recommendations=10):
        """
        Location-Based Filtering: Recommend hotels in specific destination within budget
        """
        # Normalize destination string
        destination_clean = str(destination).strip() if destination else None
        
        # Filter hotels with normalized location matching
        location_hotels = self.hotel_features[
            (self.hotel_features['location'] == destination_clean) &
            (self.hotel_features['avg_price'] >= budget_min) &
            (self.hotel_features['avg_price'] <= budget_max)
        ].sort_values('booking_count', ascending=False).head(n_recommendations)
        
        return [{'hotel_name': row['hotel_name'], 
                 'location': row['location'],
                 'avg_price': row['avg_price'],
                 'popularity': row['booking_count'],
                 'method': 'location-based'} 
                for _, row in location_hotels.iterrows()]
    
    def apply_diversity_filter(self, recommendations, diversity_factor=0.35):
        """
        Balanced diversity filter
        Ensures variety without being too aggressive
        """
        if len(recommendations) <= 1:
            return recommendations
        
        diverse_recs = [recommendations[0]]
        
        for rec in recommendations[1:]:
            # Calculate diversity penalty
            penalty = 0
            for selected in diverse_recs:
                # Location diversity - moderate penalty
                if rec['location'] == selected['location']:
                    penalty += 0.25
                
                # Price range diversity - moderate penalty
                price_diff = abs(rec['avg_price'] - selected['avg_price']) / max(rec['avg_price'], selected['avg_price'])
                if price_diff < 0.2:
                    penalty += 0.20
            
            # Adjust score
            rec['recommendation_score'] *= (1 - min(penalty, diversity_factor))
            diverse_recs.append(rec)
        
        # Re-sort after diversity adjustment
        diverse_recs = sorted(diverse_recs, key=lambda x: x['recommendation_score'], reverse=True)
        return diverse_recs

    def get_optimal_weights(self, user_code):
        """
        Simple adaptive weights based on booking history
        More bookings = trust personalization more
        """
        booking_count = self.get_user_booking_count(user_code)
        
        if booking_count == 0:
            # New users: rely on demographics
            return {
                'collaborative': 0.25,
                'content-based': 0.25,
                'gender-based': 0.50
            }
        elif booking_count < 3:
            # Occasional users: balanced approach
            return {
                'collaborative': 0.40,
                'content-based': 0.35,
                'gender-based': 0.25
            }
        else:
            # Regular users: trust collaborative filtering
            return {
                'collaborative': 0.50,
                'content-based': 0.40,
                'gender-based': 0.10
            }

    def hybrid_recommendations(self, user_code, destination=None, budget_min=0, 
                          budget_max=float('inf'), n_recommendations=10, 
                          apply_diversity=True, use_adaptive_weights=True, debug=False):
        """
        Optimized Hybrid Approach
        
        Key optimizations:
        - Simple but effective weight adaptation
        - Balanced diversity filtering
        - Efficient processing
        - Robust fallback mechanism
        
        Args:
            user_code: User identifier
            destination: Target destination (must match exact location string)
            budget_min: Minimum price filter
            budget_max: Maximum price filter
            n_recommendations: Number of recommendations to return
            apply_diversity: Whether to apply diversity filtering
            use_adaptive_weights: Adjust weights based on user experience
            debug: Enable detailed logging
            
        Returns:
            List of hotel recommendations with scores and metadata
        """
        try:
            logger.info(f"Generating recommendations for user: {user_code}")
            
            # Normalize destination filter
            filter_destination = None
            if destination:
                filter_destination = str(destination).strip()
                if debug:
                    print(f"\nDEBUG: Searching for destination: '{filter_destination}'")
                    print(f"  Available locations:")
                    for loc in self.hotel_features['location'].unique():
                        print(f"    - '{loc}'")
            
            all_recommendations = {}
            
            # Get recommendations from all methods
            collab_recs = self.collaborative_filtering_recommendations(user_code, n_recommendations * 2)
            content_recs = self.content_based_recommendations(user_code, n_recommendations * 2)
            gender_recs = self.gender_based_recommendations(user_code, n_recommendations * 2)
            
            if debug:
                booking_count = self.get_user_booking_count(user_code)
                print(f"\nDEBUG: User has {booking_count} bookings")
                print(f"DEBUG: Raw recommendations count:")
                print(f"  Collaborative: {len(collab_recs)}")
                print(f"  Content-based: {len(content_recs)}")
                print(f"  Gender-based: {len(gender_recs)}")
            
            # Normalize scores
            def normalize_scores(recs):
                if not recs:
                    return []
                max_score = max(rec['score'] for rec in recs)
                if max_score == 0:
                    return recs
                for rec in recs:
                    rec['score'] = rec['score'] / max_score
                return recs
            
            collab_recs = normalize_scores(collab_recs)
            content_recs = normalize_scores(content_recs)
            gender_recs = normalize_scores(gender_recs)
            
            # Get weights
            if use_adaptive_weights:
                weights = self.get_optimal_weights(user_code)
                if debug:
                    print(f"DEBUG: Using adaptive weights: {weights}")
            else:
                weights = {
                    'collaborative': 0.50,
                    'content-based': 0.40,
                    'gender-based': 0.10
                }
            
            # Combine recommendations
            for rec in collab_recs + content_recs + gender_recs:
                hotel_name = rec['hotel_name']
                method = rec['method']
                score = rec['score']
                
                if hotel_name not in all_recommendations:
                    all_recommendations[hotel_name] = {'total_score': 0, 'methods': []}
                
                all_recommendations[hotel_name]['total_score'] += score * weights[method]
                all_recommendations[hotel_name]['methods'].append(method)
            
            # Apply filters and build final recommendations
            final_recommendations = []
            for hotel_name, data in all_recommendations.items():
                hotel_info = self.hotel_features[self.hotel_features['hotel_name'] == hotel_name]
                if len(hotel_info) > 0:
                    hotel_info = hotel_info.iloc[0]
                    
                    # Get normalized hotel location
                    hotel_location = str(hotel_info['location']).strip()
                    
                    # Check destination filter
                    if filter_destination and hotel_location != filter_destination:
                        if debug:
                            print(f"DEBUG: Filtered out {hotel_name} - Location mismatch")
                        continue
                    
                    # Check budget filter
                    hotel_price = float(hotel_info['avg_price'])
                    if hotel_price < budget_min or hotel_price > budget_max:
                        if debug:
                            print(f"DEBUG: Filtered out {hotel_name} - Price ${hotel_price:.2f} outside budget")
                        continue
                    
                    # Passed all filters
                    if debug:
                        print(f"DEBUG: Included {hotel_name} - {hotel_location}, ${hotel_price:.2f}")
                    
                    final_recommendations.append({
                        'hotel_name': hotel_name,
                        'location': hotel_location,
                        'avg_price': round(hotel_price, 2),
                        'avg_stay': round(hotel_info['avg_stay'], 2),
                        'popularity': int(hotel_info['booking_count']),
                        'recommendation_score': round(data['total_score'], 4),
                        'methods_used': ', '.join(set(data['methods']))
                    })
            
            # Sort by score
            final_recommendations = sorted(final_recommendations, 
                                        key=lambda x: x['recommendation_score'], 
                                        reverse=True)
            
            # Apply balanced diversity filter
            if apply_diversity and len(final_recommendations) > 1:
                final_recommendations = self.apply_diversity_filter(final_recommendations)
            
            # Limit to requested number
            final_recommendations = final_recommendations[:n_recommendations]
            
            # Fallback mechanism
            if not final_recommendations:
                logger.warning(f"No hybrid recommendations found for user {user_code}. Using fallback.")
                if debug:
                    print(f"\nDEBUG: No hybrid recommendations. Applying fallback...")
                
                matching_hotels = self.hotel_features.copy()
                
                # Apply destination filter
                if filter_destination:
                    matching_hotels = matching_hotels[matching_hotels['location'] == filter_destination]
                    if debug:
                        print(f"  Hotels in '{filter_destination}': {len(matching_hotels)}")
                
                # Apply budget filter
                matching_hotels = matching_hotels[
                    (matching_hotels['avg_price'] >= budget_min) &
                    (matching_hotels['avg_price'] <= budget_max)
                ]
                
                # Sort by popularity
                matching_hotels = matching_hotels.sort_values('booking_count', ascending=False).head(n_recommendations)
                
                final_recommendations = [{
                    'hotel_name': row['hotel_name'],
                    'location': row['location'],
                    'avg_price': round(row['avg_price'], 2),
                    'avg_stay': round(row['avg_stay'], 2),
                    'popularity': int(row['booking_count']),
                    'recommendation_score': round(row['booking_count'] / self.hotel_features['booking_count'].max(), 4),
                    'methods_used': 'popularity-based (fallback)'
                } for _, row in matching_hotels.iterrows()]
            
            logger.info(f"Generated {len(final_recommendations)} recommendations for user {user_code}")
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_code}: {e}")
            raise
    
    def get_recommendations_summary(self, user_code, n_recommendations=10):
        """
        Generate comprehensive recommendation summary for a user
        """
        user_info = self.get_user_info(user_code)
        if not user_info:
            return None
        
        recommendations = self.hybrid_recommendations(user_code, n_recommendations=n_recommendations)
        
        # Calculate user statistics
        user_bookings = self.complete_df[self.complete_df['userCode'] == user_code]
        
        summary = {
            'user_info': user_info,
            'user_stats': {
                'total_bookings': len(user_bookings),
                'avg_price': user_bookings['price'].mean() if len(user_bookings) > 0 else 0,
                'avg_stay': user_bookings['days'].mean() if len(user_bookings) > 0 else 0,
                'favorite_locations': user_bookings['place'].value_counts().head(3).to_dict() if len(user_bookings) > 0 else {}
            },
            'recommendations': recommendations
        }
        
        return summary


# ============================================================================
# OPTIMIZED MODEL LOADING UTILITY
# ============================================================================

def load_recommendation_models(models_dir='models'):
    """
    Load only the essential models needed for hotel recommendations
    
    Args:
        models_dir (str): Directory containing the saved model files
    
    Returns:
        dict: Dictionary containing all required models
    """
    import pickle
    import os
    
    models = {}
    
    required_files = [
        'user_hotel_matrix.pkl',
        'user_similarity.pkl',
        'hotel_similarity.pkl',
        'hotel_features.pkl',
        'complete_data.pkl',
        'users_data.pkl'
    ]
    
    logger.info("Loading recommendation models...")
    print("Loading recommendation models...")
    print("-" * 70)
    
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            error_msg = f"Required file not found: {filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        with open(filepath, 'rb') as f:
            model_name = filename.replace('.pkl', '')
            models[model_name] = pickle.load(f)
        
        print(f"  Loaded: {filename}")
        logger.info(f"Loaded: {filename}")
    
    print("-" * 70)
    print("All models loaded successfully!")
    logger.info("All models loaded successfully!")
    
    return models