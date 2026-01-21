"""
Professional Recommendation System Evaluation Module
Adds missing evaluation metrics and validation framework
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    """
    Evaluates recommendation system using industry-standard metrics
    """
    
    def __init__(self, complete_df, user_hotel_matrix):
        self.complete_df = complete_df
        self.user_hotel_matrix = user_hotel_matrix
        self.test_interactions = None
        self.train_interactions = None
        
    def create_train_test_split(self, test_size=0.2, min_interactions=3):
        """
        Split user-hotel interactions into train/test sets
        Only include users with minimum interactions
        """
        print(f"\n{'='*70}")
        print("CREATING TRAIN/TEST SPLIT")
        print(f"{'='*70}")
        
        # Get user interaction counts
        user_interactions = self.user_hotel_matrix.apply(lambda x: (x > 0).sum(), axis=1)
        valid_users = user_interactions[user_interactions >= min_interactions].index
        
        print(f"Total users: {len(self.user_hotel_matrix)}")
        print(f"Users with {min_interactions}+ interactions: {len(valid_users)}")
        
        train_data = []
        test_data = []
        
        for user in valid_users:
            # Get hotels this user interacted with
            user_hotels = self.user_hotel_matrix.loc[user]
            interacted_hotels = user_hotels[user_hotels > 0].index.tolist()
            
            if len(interacted_hotels) >= min_interactions:
                # Split into train/test
                train_hotels, test_hotels = train_test_split(
                    interacted_hotels, 
                    test_size=test_size,
                    random_state=42
                )
                
                for hotel in train_hotels:
                    train_data.append({
                        'userCode': user,
                        'hotel_name': hotel,
                        'rating': user_hotels[hotel]
                    })
                
                for hotel in test_hotels:
                    test_data.append({
                        'userCode': user,
                        'hotel_name': hotel,
                        'rating': user_hotels[hotel]
                    })
        
        self.train_interactions = pd.DataFrame(train_data)
        self.test_interactions = pd.DataFrame(test_data)
        
        print(f"\nTrain interactions: {len(self.train_interactions)}")
        print(f"Test interactions: {len(self.test_interactions)}")
        print(f"Split ratio: {len(self.test_interactions) / (len(self.train_interactions) + len(self.test_interactions)):.2%}")
        
        return self.train_interactions, self.test_interactions
    
    def precision_at_k(self, recommendations, actual_items, k=5):
        """
        Precision@K: What proportion of recommended items are relevant?
        """
        if not recommendations or not actual_items:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_recs = [rec for rec in top_k_recs if rec in actual_items]
        
        return len(relevant_recs) / k
    
    def recall_at_k(self, recommendations, actual_items, k=5):
        """
        Recall@K: What proportion of relevant items were recommended?
        """
        if not recommendations or not actual_items:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_recs = [rec for rec in top_k_recs if rec in actual_items]
        
        return len(relevant_recs) / len(actual_items) if actual_items else 0.0
    
    def mean_average_precision(self, recommendations, actual_items, k=10):
        """
        MAP@K: Mean Average Precision
        """
        if not recommendations or not actual_items:
            return 0.0
        
        top_k_recs = recommendations[:k]
        score = 0.0
        num_hits = 0.0
        
        for i, rec in enumerate(top_k_recs):
            if rec in actual_items:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(actual_items), k) if actual_items else 0.0
    
    def ndcg_at_k(self, recommendations, actual_items, k=10):
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Considers ranking quality
        """
        if not recommendations or not actual_items:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k]):
            if rec in actual_items:
                dcg += 1.0 / np.log2(i + 2)  # +2 because index starts at 0
        
        # IDCG (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(self, recommendations, actual_items, k=5):
        """
        Hit Rate@K: Did we recommend at least one relevant item?
        """
        if not recommendations or not actual_items:
            return 0.0
        
        top_k_recs = set(recommendations[:k])
        return 1.0 if len(top_k_recs.intersection(set(actual_items))) > 0 else 0.0
    
    def coverage(self, all_recommendations, total_items):
        """
        Catalog Coverage: What % of items were ever recommended?
        """
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / total_items if total_items > 0 else 0.0
    
    def diversity(self, recommendations):
        """
        Recommendation Diversity: How different are recommendations?
        Based on location and price variance
        """
        if len(recommendations) <= 1:
            return 0.0
        
        locations = [rec.get('location', '') for rec in recommendations]
        prices = [rec.get('avg_price', 0) for rec in recommendations]
        
        # Location diversity
        unique_locations = len(set(locations))
        location_diversity = unique_locations / len(recommendations)
        
        # Price diversity (coefficient of variation)
        price_diversity = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        return (location_diversity + min(price_diversity, 1.0)) / 2
    
    def evaluate_model(self, rec_engine, k_values=[3, 5, 10], sample_users=50):
        """
        Comprehensive model evaluation
        """
        if self.test_interactions is None:
            print("ERROR: Must run create_train_test_split() first!")
            return None
        
        print(f"\n{'='*70}")
        print("EVALUATING RECOMMENDATION MODEL")
        print(f"{'='*70}")
        
        # Get unique test users
        test_users = self.test_interactions['userCode'].unique()
        
        # Sample users if too many
        if len(test_users) > sample_users:
            test_users = np.random.choice(test_users, sample_users, replace=False)
            print(f"Evaluating on {sample_users} sampled users")
        else:
            print(f"Evaluating on {len(test_users)} users")
        
        results = defaultdict(list)
        all_recommendations = []
        
        for user in test_users:
            # Get actual test items for this user
            actual_items = self.test_interactions[
                self.test_interactions['userCode'] == user
            ]['hotel_name'].tolist()
            
            # Get recommendations
            try:
                recs = rec_engine.hybrid_recommendations(user, n_recommendations=max(k_values))
                rec_hotels = [rec['hotel_name'] for rec in recs]
                all_recommendations.append(rec_hotels)
                
                # Calculate metrics for each k
                for k in k_values:
                    results[f'precision@{k}'].append(
                        self.precision_at_k(rec_hotels, actual_items, k)
                    )
                    results[f'recall@{k}'].append(
                        self.recall_at_k(rec_hotels, actual_items, k)
                    )
                    results[f'ndcg@{k}'].append(
                        self.ndcg_at_k(rec_hotels, actual_items, k)
                    )
                    results[f'hit_rate@{k}'].append(
                        self.hit_rate_at_k(rec_hotels, actual_items, k)
                    )
                
                results['map'].append(
                    self.mean_average_precision(rec_hotels, actual_items, max(k_values))
                )
                results['diversity'].append(
                    self.diversity(recs)
                )
                
            except Exception as e:
                print(f"Error evaluating user {user}: {e}")
                continue
        
        # Calculate averages
        avg_results = {metric: np.mean(values) for metric, values in results.items()}
        
        # Calculate coverage
        total_hotels = self.user_hotel_matrix.shape[1]
        avg_results['coverage'] = self.coverage(all_recommendations, total_hotels)
        
        return avg_results
    
    def print_evaluation_results(self, results):
        """
        Pretty print evaluation results
        """
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        
        print("\n ACCURACY METRICS:")
        print("-" * 70)
        for metric, value in sorted(results.items()):
            if '@' in metric or metric == 'map':
                print(f"  {metric.upper():<20}: {value:.4f}")
        
        print("\n BUSINESS METRICS:")
        print("-" * 70)
        print(f"  {'Coverage':<20}: {results['coverage']:.4f} ({results['coverage']*100:.2f}%)")
        print(f"  {'Diversity':<20}: {results['diversity']:.4f}")
        
        print("\n INTERPRETATION:")
        print("-" * 70)
        
        # Precision interpretation
        precision_5 = results.get('precision@5', 0)
        if precision_5 > 0.3:
            print(f"Precision@5 ({precision_5:.2%}) - GOOD: Recommendations are relevant")
        elif precision_5 > 0.15:
            print(f"Precision@5 ({precision_5:.2%}) - FAIR: Room for improvement")
        else:
            print(f"Precision@5 ({precision_5:.2%}) - POOR: Need better algorithms")
        
        # Coverage interpretation
        if results['coverage'] > 0.5:
            print(f"Coverage ({results['coverage']:.2%}) - GOOD: Broad catalog exposure")
        else:
            print(f"Coverage ({results['coverage']:.2%}) - LOW: Too focused on few items")
        
        # Diversity interpretation
        if results['diversity'] > 0.5:
            print(f"Diversity ({results['diversity']:.2f}) - GOOD: Varied recommendations")
        else:
            print(f"Diversity ({results['diversity']:.2f}) - LOW: Recommendations too similar")
    
    def plot_evaluation_results(self, results):
        """
        Visualize evaluation metrics
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Precision/Recall/NDCG at different K
        k_metrics = {}
        for metric, value in results.items():
            if '@' in metric:
                k = int(metric.split('@')[1])
                metric_name = metric.split('@')[0]
                if k not in k_metrics:
                    k_metrics[k] = {}
                k_metrics[k][metric_name] = value
        
        k_values = sorted(k_metrics.keys())
        precision_vals = [k_metrics[k].get('precision', 0) for k in k_values]
        recall_vals = [k_metrics[k].get('recall', 0) for k in k_values]
        ndcg_vals = [k_metrics[k].get('ndcg', 0) for k in k_values]
        
        axes[0].plot(k_values, precision_vals, marker='o', label='Precision', linewidth=2)
        axes[0].plot(k_values, recall_vals, marker='s', label='Recall', linewidth=2)
        axes[0].plot(k_values, ndcg_vals, marker='^', label='NDCG', linewidth=2)
        axes[0].set_xlabel('K (Number of Recommendations)', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Metrics vs K', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Hit Rate
        hit_rates = [k_metrics[k].get('hit_rate', 0) for k in k_values]
        axes[1].bar(k_values, hit_rates, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('K', fontsize=12)
        axes[1].set_ylabel('Hit Rate', fontsize=12)
        axes[1].set_title('Hit Rate@K', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Coverage & Diversity
        business_metrics = ['coverage', 'diversity', 'map']
        business_values = [results.get(m, 0) for m in business_metrics]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        axes[2].bar(business_metrics, business_values, color=colors, alpha=0.7)
        axes[2].set_ylabel('Score', fontsize=12)
        axes[2].set_title('Business Metrics', fontsize=14, fontweight='bold')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('../outputs/evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Evaluation plot saved to '../outputs/evaluation_metrics.png'")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_evaluation(rec_engine, complete_df, user_hotel_matrix):
    """
    Complete evaluation pipeline
    """
    # Initialize evaluator
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
        sample_users=50
    )
    
    # Print results
    evaluator.print_evaluation_results(results)
    
    # Visualize results
    evaluator.plot_evaluation_results(results)
    
    return results, evaluator


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# Add this to your main code after initializing rec_engine:

print("\n" + "="*70)
print("RUNNING PROFESSIONAL EVALUATION")
print("="*70)

results, evaluator = run_evaluation(rec_engine, complete_df, user_hotel_matrix)

# Save evaluation results
import json
with open('../outputs/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Evaluation complete!")
print("✓ Results saved to '../outputs/evaluation_results.json'")
"""