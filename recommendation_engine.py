import glob
import os
import joblib
import pandas as pd
import numpy as np
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
        # ✅ CRITICAL FIX: Handle None values safely
        self.user_hotel_matrix = user_hotel_matrix if user_hotel_matrix is not None else pd.DataFrame()
        self.user_similarity_df = user_similarity_df if user_similarity_df is not None else pd.DataFrame()
        self.hotel_similarity_df = hotel_similarity_df if hotel_similarity_df is not None else pd.DataFrame()
        
        # ✅ CRITICAL FIX: Safe copy with None check
        if hotel_features is not None:
            self.hotel_features = hotel_features.copy()
            # Normalize hotel_features locations on initialization
            self.hotel_features['location'] = self.hotel_features['location'].astype(str).str.strip()
        else:
            self.hotel_features = pd.DataFrame(columns=['hotel_name', 'location', 'avg_price', 'avg_stay', 'booking_count'])
        
        self.complete_df = complete_df if complete_df is not None else pd.DataFrame()
        self.users_df = users_df if users_df is not None else pd.DataFrame(columns=['code', 'name', 'age', 'gender', 'company'])
        
        # CRITICAL FIX: Ensure user codes are consistent type across all dataframes
        self._normalize_user_codes()
        
        logger.info("HotelRecommendationEngine initialized successfully")
    
    def _normalize_user_codes(self):
        """Normalize user codes to integers across all data structures"""
        try:
            # ✅ Only normalize if dataframes have data
            if not self.users_df.empty and 'code' in self.users_df.columns:
                self.users_df['code'] = self.users_df['code'].astype(int)
            
            # Convert user codes in complete_df
            if not self.complete_df.empty and 'userCode' in self.complete_df.columns:
                self.complete_df['userCode'] = self.complete_df['userCode'].astype(int)
            
            # Convert user_hotel_matrix index
            if not self.user_hotel_matrix.empty:
                self.user_hotel_matrix.index = self.user_hotel_matrix.index.astype(int)
            
            # Convert similarity matrix indices
            if not self.user_similarity_df.empty:
                self.user_similarity_df.index = self.user_similarity_df.index.astype(int)
                self.user_similarity_df.columns = self.user_similarity_df.columns.astype(int)
            
            logger.info("User codes normalized to integers across all data structures")
        except Exception as e:
            logger.warning(f"Could not normalize user codes: {e}")
    
    def _convert_user_code(self, user_code):
        """Convert user_code to integer for consistent lookup"""
        try:
            return int(user_code)
        except (ValueError, TypeError):
            logger.error(f"Cannot convert user_code to int: {user_code}")
            return user_code
        
    def get_user_info(self, user_code):
        """Get user demographic information"""
        user_code = self._convert_user_code(user_code)
        
        if self.users_df.empty:
            logger.warning("Users dataframe is empty")
            return None
            
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
    
    def get_user_stats(self, user_code):
        """Get statistical summary of user's booking history"""
        user_code = self._convert_user_code(user_code)
        
        if self.complete_df.empty:
            return {
                'total_bookings': 0,
                'avg_price': 0.0,
                'avg_stay': 0.0,
                'favorite_locations': {}
            }
            
        user_bookings = self.complete_df[self.complete_df['userCode'] == user_code]
        
        if len(user_bookings) == 0:
            return {
                'total_bookings': 0,
                'avg_price': 0.0,
                'avg_stay': 0.0,
                'favorite_locations': {}
            }
        
        total_bookings = len(user_bookings)
        avg_price = user_bookings['price'].mean() if 'price' in user_bookings.columns else 0.0
        avg_stay = user_bookings['days'].mean() if 'days' in user_bookings.columns else 0.0
        
        if 'location' in user_bookings.columns:
            location_counts = user_bookings['location'].value_counts().to_dict()
        elif 'place' in user_bookings.columns:
            location_counts = user_bookings['place'].value_counts().to_dict()
        else:
            location_counts = {}
        
        return {
            'total_bookings': total_bookings,
            'avg_price': float(avg_price),
            'avg_stay': float(avg_stay),
            'favorite_locations': location_counts
        }

    def get_user_booking_count(self, user_code):
        """Get number of bookings for experience-based adjustments"""
        user_code = self._convert_user_code(user_code)
        
        if self.complete_df.empty:
            return 0
            
        user_bookings = self.complete_df[self.complete_df['userCode'] == user_code]
        return len(user_bookings)
    
    def validate_inputs(self, user_code=None, destination=None, budget_min=0, budget_max=float('inf')):
        """Validate all inputs before processing"""
        
        if budget_min < 0:
            raise ValueError("budget_min cannot be negative")
        
        if budget_max < budget_min:
            raise ValueError("budget_max must be >= budget_min")
        
        if budget_max > 500:
            raise ValueError("budget_max seems unreasonably high. Hotel prices range from ₹60 to ₹313")
        
        if budget_min < 50:
            raise ValueError("budget_min seems unreasonably low. Minimum hotel price is ₹60")
        
        if destination and not self.hotel_features.empty:
            destination_normalized = str(destination).strip()
            valid_locations = self.hotel_features['location'].unique()
            if destination_normalized not in valid_locations:
                raise ValueError(f"Invalid destination. Must be one of: {list(valid_locations)}")
        
        logger.debug(f"Input validation passed - user: {user_code}, destination: {destination}, budget: ${budget_min}-${budget_max}")
        return True
    
    def collaborative_filtering_recommendations(self, user_code, n_recommendations=10, allow_visited=False):
        """Collaborative Filtering - Returns empty list for new users"""
        user_code = self._convert_user_code(user_code)
        
        if self.user_similarity_df.empty or self.user_hotel_matrix.empty:
            return []
        
        if user_code not in self.user_similarity_df.index:
            return []
        
        similar_users = self.user_similarity_df[user_code].sort_values(ascending=False)[1:11]
        user_hotels = set(self.user_hotel_matrix.loc[user_code][self.user_hotel_matrix.loc[user_code] > 0].index)
        
        recommendations = {}
        for similar_user_code, similarity_score in similar_users.items():
            if similar_user_code in self.user_hotel_matrix.index:
                similar_user_hotels = self.user_hotel_matrix.loc[similar_user_code]
                for hotel, rating in similar_user_hotels.items():
                    if rating > 0:
                        if allow_visited or hotel not in user_hotels:
                            if hotel not in recommendations:
                                recommendations[hotel] = 0
                            recommendations[hotel] += rating * similarity_score
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        return [{'hotel_name': hotel, 'score': score, 'method': 'collaborative'} for hotel, score in sorted_recs]
    
    def content_based_recommendations(self, user_code, n_recommendations=10, allow_visited=False):
        """Content-Based Filtering"""
        user_code = self._convert_user_code(user_code)
        
        if self.user_hotel_matrix.empty or self.hotel_similarity_df.empty:
            return []
        
        if user_code not in self.user_hotel_matrix.index:
            return []
        
        user_hotels = self.user_hotel_matrix.loc[user_code]
        liked_hotels = user_hotels[user_hotels > 0].index.tolist()
        
        if not liked_hotels:
            return []
        
        recommendations = {}
        for hotel in liked_hotels:
            if hotel in self.hotel_similarity_df.index:
                similar_hotels = self.hotel_similarity_df[hotel].sort_values(ascending=False)[1:6]
                for similar_hotel, similarity_score in similar_hotels.items():
                    if allow_visited or similar_hotel not in liked_hotels:
                        if similar_hotel not in recommendations:
                            recommendations[similar_hotel] = 0
                        recommendations[similar_hotel] += similarity_score * user_hotels[hotel]
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        return [{'hotel_name': hotel, 'score': score, 'method': 'content-based'} for hotel, score in sorted_recs]
    
    def gender_based_recommendations(self, user_code, n_recommendations=10, allow_visited=False):
        """Gender-Based Filtering with Age Grouping"""
        user_code = self._convert_user_code(user_code)
        user_info = self.get_user_info(user_code)
        
        if not user_info or self.complete_df.empty:
            return []
        
        user_gender = user_info['gender']
        user_age = user_info['age']
        
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
        
        gender_age_df = self.complete_df[
            (self.complete_df['gender'] == user_gender) &
            (self.complete_df['age'] >= age_range[0]) &
            (self.complete_df['age'] <= age_range[1])
        ]
        
        if len(gender_age_df) == 0:
            gender_age_df = self.complete_df[self.complete_df['gender'] == user_gender]
        
        if len(gender_age_df) == 0:
            return []
            
        hotel_popularity = gender_age_df.groupby('hotel_name').agg({
            'userCode': 'count',
            'price': 'mean',
            'days': 'mean'
        }).reset_index()
        hotel_popularity.columns = ['hotel_name', 'popularity', 'avg_price', 'avg_days']
        
        if not allow_visited and not self.user_hotel_matrix.empty and user_code in self.user_hotel_matrix.index:
            user_hotels = set(self.user_hotel_matrix.loc[user_code][self.user_hotel_matrix.loc[user_code] > 0].index)
            hotel_popularity = hotel_popularity[~hotel_popularity['hotel_name'].isin(user_hotels)]
        
        hotel_popularity = hotel_popularity.sort_values('popularity', ascending=False).head(n_recommendations)
        
        return [{'hotel_name': row['hotel_name'], 'score': row['popularity'], 'method': 'gender-based', 'avg_price': row['avg_price'], 'age_group': age_group} for _, row in hotel_popularity.iterrows()]
    
    def apply_diversity_filter(self, recommendations, diversity_factor=0.35):
        """Balanced diversity filter"""
        if len(recommendations) <= 1:
            return recommendations
        
        diverse_recs = [recommendations[0]]
        
        for rec in recommendations[1:]:
            penalty = 0
            for selected in diverse_recs:
                if rec.get('location') == selected.get('location'):
                    penalty += 0.25
                
                rec_price = rec.get('avg_price', 0)
                sel_price = selected.get('avg_price', 0)
                if rec_price > 0 and sel_price > 0:
                    price_diff = abs(rec_price - sel_price) / max(rec_price, sel_price)
                    if price_diff < 0.2:
                        penalty += 0.20
            
            rec['recommendation_score'] *= (1 - min(penalty, diversity_factor))
            diverse_recs.append(rec)
        
        diverse_recs = sorted(diverse_recs, key=lambda x: x['recommendation_score'], reverse=True)
        return diverse_recs

    def get_optimal_weights(self, user_code):
        """Simple adaptive weights based on booking history"""
        user_code = self._convert_user_code(user_code)
        booking_count = self.get_user_booking_count(user_code)
        
        if booking_count == 0:
            return {'collaborative': 0.25, 'content-based': 0.25, 'gender-based': 0.50}
        elif booking_count < 3:
            return {'collaborative': 0.40, 'content-based': 0.35, 'gender-based': 0.25}
        else:
            return {'collaborative': 0.50, 'content-based': 0.40, 'gender-based': 0.10}

    def hybrid_recommendations(self, user_code, destination=None, budget_min=0, budget_max=float('inf'), n_recommendations=10, apply_diversity=True, use_adaptive_weights=True, debug=False):
        """Hybrid recommendation system with fallback"""
        try:
            user_code = self._convert_user_code(user_code)
            
            if self.hotel_features.empty:
                logger.error("No hotel data available")
                return []
            
            is_new_user = self.user_hotel_matrix.empty or user_code not in self.user_hotel_matrix.index
            filter_destination = str(destination).strip() if destination else None
            
            all_recommendations = {}
            
            collab_recs = self.collaborative_filtering_recommendations(user_code, n_recommendations * 2, allow_visited=False)
            content_recs = self.content_based_recommendations(user_code, n_recommendations * 2, allow_visited=False)
            gender_recs = self.gender_based_recommendations(user_code, n_recommendations * 2, allow_visited=False)
            
            total_recs_first_attempt = len(collab_recs) + len(content_recs) + len(gender_recs)
            
            if total_recs_first_attempt == 0:
                collab_recs = self.collaborative_filtering_recommendations(user_code, n_recommendations * 2, allow_visited=True)
                content_recs = self.content_based_recommendations(user_code, n_recommendations * 2, allow_visited=True)
                gender_recs = self.gender_based_recommendations(user_code, n_recommendations * 2, allow_visited=True)
            
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
            
            if use_adaptive_weights:
                if is_new_user:
                    weights = {'collaborative': 0.0, 'content-based': 0.0, 'gender-based': 1.0}
                else:
                    weights = self.get_optimal_weights(user_code)
            else:
                weights = {'collaborative': 0.50, 'content-based': 0.40, 'gender-based': 0.10}
            
            for rec in collab_recs + content_recs + gender_recs:
                hotel_name = rec['hotel_name']
                method = rec['method']
                score = rec['score']
                
                if hotel_name not in all_recommendations:
                    all_recommendations[hotel_name] = {'total_score': 0, 'methods': []}
                
                all_recommendations[hotel_name]['total_score'] += score * weights[method]
                all_recommendations[hotel_name]['methods'].append(method)
            
            final_recommendations = []
            for hotel_name, data in all_recommendations.items():
                hotel_info = self.hotel_features[self.hotel_features['hotel_name'] == hotel_name]
                if len(hotel_info) > 0:
                    hotel_info = hotel_info.iloc[0]
                    hotel_location = str(hotel_info['location']).strip()
                    
                    if filter_destination and hotel_location != filter_destination:
                        continue
                    
                    hotel_price = float(hotel_info['avg_price'])
                    if hotel_price < budget_min or hotel_price > budget_max:
                        continue
                    
                    final_recommendations.append({
                        'hotel_name': hotel_name,
                        'location': hotel_location,
                        'avg_price': round(hotel_price, 2),
                        'avg_stay': round(float(hotel_info['avg_stay']), 2),
                        'popularity': int(hotel_info['booking_count']),
                        'recommendation_score': round(data['total_score'], 4),
                        'methods_used': ', '.join(set(data['methods']))
                    })
            
            final_recommendations = sorted(final_recommendations, key=lambda x: x['recommendation_score'], reverse=True)
            
            if apply_diversity and len(final_recommendations) > 1:
                final_recommendations = self.apply_diversity_filter(final_recommendations)
            
            final_recommendations = final_recommendations[:n_recommendations]
            
            if not final_recommendations:
                matching_hotels = self.hotel_features.copy()
                
                if filter_destination:
                    matching_hotels = matching_hotels[matching_hotels['location'] == filter_destination]
                
                matching_hotels = matching_hotels[
                    (matching_hotels['avg_price'] >= budget_min) &
                    (matching_hotels['avg_price'] <= budget_max)
                ]
                
                matching_hotels = matching_hotels.sort_values('booking_count', ascending=False).head(n_recommendations)
                
                max_popularity = self.hotel_features['booking_count'].max() if not self.hotel_features.empty else 1
                final_recommendations = [{
                    'hotel_name': row['hotel_name'],
                    'location': row['location'],
                    'avg_price': round(float(row['avg_price']), 2),
                    'avg_stay': round(float(row['avg_stay']), 2),
                    'popularity': int(row['booking_count']),
                    'recommendation_score': round(float(row['booking_count']) / max_popularity, 4),
                    'methods_used': 'popularity-based (fallback)'
                } for _, row in matching_hotels.iterrows()]
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            raise


def load_recommendation_models(models_dir='models/recommendation'):
    """Load recommendation models from directory"""
    models = {}
    
    if not os.path.exists(models_dir):
        print(f"⚠️  WARNING: Models directory '{models_dir}' not found")
        return models
    
    try:
        # Look for both .joblib and .pkl files
        model_files = glob.glob(os.path.join(models_dir, '*.joblib')) + glob.glob(os.path.join(models_dir, '*.pkl'))

        if not model_files:
            print(f"⚠️  WARNING: No .pkl or .joblib files found in '{models_dir}'")
            return models
        
        print(f"📂 Found {len(model_files)} model files")
        
        for filepath in model_files:
            model_name = os.path.basename(filepath)
            if model_name.endswith('.pkl'):
                model_name = model_name[:-4]
            elif model_name.endswith('.joblib'):
                model_name = model_name[:-7]
            model_name = model_name.strip()

            try:
                models[model_name] = joblib.load(filepath)
                print(f"  ✅ Loaded: {model_name}")
            except Exception as e:
                print(f"  ⚠️  Failed to load {model_name}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return {}
    
    print(f"✅ Successfully loaded {len(models)} models")
    return models