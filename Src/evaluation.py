"""
Professional Recommendation System Evaluation Module
Industry-standard metrics with proper validation and error handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class RecommendationEvaluator:
    """
    Evaluates recommendation systems using industry-standard metrics
    Handles edge cases and provides comprehensive validation
    """
    
    def __init__(self, complete_df: pd.DataFrame, user_hotel_matrix: pd.DataFrame):
        """
        Initialize evaluator
        
        Args:
            complete_df: Complete interaction data
            user_hotel_matrix: User-hotel interaction matrix (users × hotels)
        """
        self.complete_df = complete_df.copy()
        self.user_hotel_matrix = user_hotel_matrix.copy()
        self.test_interactions = None
        self.train_interactions = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data"""
        if self.complete_df is None or len(self.complete_df) == 0:
            raise ValueError("complete_df cannot be empty")
        
        if self.user_hotel_matrix is None or self.user_hotel_matrix.shape[0] == 0:
            raise ValueError("user_hotel_matrix cannot be empty")
        
        required_cols = ['userCode', 'hotel_name']
        missing_cols = [col for col in required_cols if col not in self.complete_df.columns]
        if missing_cols:
            raise ValueError(f"complete_df missing required columns: {missing_cols}")
    
    def create_train_test_split(
        self, 
        test_size: float = 0.2, 
        min_interactions: int = 3,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split user-hotel interactions into train/test sets
        Only includes users with minimum interactions to ensure valid test sets
        
        Args:
            test_size: Proportion of interactions to use for testing (0-1)
            min_interactions: Minimum interactions required per user
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_interactions, test_interactions) DataFrames
        """
        print(f"\n{'='*70}")
        print("CREATING TRAIN/TEST SPLIT")
        print(f"{'='*70}")
        
        # Validate parameters
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if min_interactions < 2:
            raise ValueError("min_interactions must be at least 2")
        
        # Get user interaction counts
        user_interactions = self.user_hotel_matrix.apply(lambda x: (x > 0).sum(), axis=1)
        valid_users = user_interactions[user_interactions >= min_interactions].index
        
        print(f"Total users: {len(self.user_hotel_matrix)}")
        print(f"Users with {min_interactions}+ interactions: {len(valid_users)}")
        
        if len(valid_users) == 0:
            raise ValueError(f"No users have {min_interactions}+ interactions. Lower min_interactions.")
        
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
                    random_state=random_state
                )
                
                # Ensure at least one item in each split
                if len(train_hotels) == 0 or len(test_hotels) == 0:
                    continue
                
                for hotel in train_hotels:
                    train_data.append({
                        'userCode': user,
                        'hotel_name': hotel,
                        'rating': float(user_hotels[hotel])
                    })
                
                for hotel in test_hotels:
                    test_data.append({
                        'userCode': user,
                        'hotel_name': hotel,
                        'rating': float(user_hotels[hotel])
                    })
        
        self.train_interactions = pd.DataFrame(train_data)
        self.test_interactions = pd.DataFrame(test_data)
        
        # Validate split
        if len(self.train_interactions) == 0 or len(self.test_interactions) == 0:
            raise ValueError("Train/test split resulted in empty set. Adjust parameters.")
        
        actual_test_ratio = len(self.test_interactions) / (
            len(self.train_interactions) + len(self.test_interactions)
        )
        
        print(f"\nTrain interactions: {len(self.train_interactions):,}")
        print(f"Test interactions: {len(self.test_interactions):,}")
        print(f"Actual split ratio: {actual_test_ratio:.2%} (target: {test_size:.2%})")
        
        return self.train_interactions, self.test_interactions
    
    def precision_at_k(
        self, 
        recommendations: List[str], 
        actual_items: List[str], 
        k: int = 5
    ) -> float:
        """
        Precision@K: Proportion of recommended items that are relevant
        
        Formula: |recommended ∩ relevant| / k
        
        Args:
            recommendations: List of recommended item IDs
            actual_items: List of actual relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision score (0-1)
        """
        if not recommendations or not actual_items or k <= 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_recs = [rec for rec in top_k_recs if rec in actual_items]
        
        return len(relevant_recs) / k
    
    def recall_at_k(
        self, 
        recommendations: List[str], 
        actual_items: List[str], 
        k: int = 5
    ) -> float:
        """
        Recall@K: Proportion of relevant items that were recommended
        
        Formula: |recommended ∩ relevant| / |relevant|
        
        Args:
            recommendations: List of recommended item IDs
            actual_items: List of actual relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall score (0-1)
        """
        if not recommendations or not actual_items or k <= 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_recs = [rec for rec in top_k_recs if rec in actual_items]
        
        return len(relevant_recs) / len(actual_items)
    
    def f1_score_at_k(
        self, 
        recommendations: List[str], 
        actual_items: List[str], 
        k: int = 5
    ) -> float:
        """
        F1@K: Harmonic mean of Precision@K and Recall@K
        
        Args:
            recommendations: List of recommended item IDs
            actual_items: List of actual relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            F1 score (0-1)
        """
        precision = self.precision_at_k(recommendations, actual_items, k)
        recall = self.recall_at_k(recommendations, actual_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_average_precision(
        self, 
        recommendations: List[str], 
        actual_items: List[str], 
        k: int = 10
    ) -> float:
        """
        MAP@K: Mean Average Precision
        Rewards placing relevant items higher in the ranking
        
        Args:
            recommendations: List of recommended item IDs
            actual_items: List of actual relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP score (0-1)
        """
        if not recommendations or not actual_items or k <= 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        score = 0.0
        num_hits = 0.0
        
        for i, rec in enumerate(top_k_recs):
            if rec in actual_items:
                num_hits += 1.0
                # Precision at position i+1
                score += num_hits / (i + 1.0)
        
        # Normalize by minimum of k and number of relevant items
        return score / min(len(actual_items), k)
    
    def ndcg_at_k(
        self, 
        recommendations: List[str], 
        actual_items: List[str], 
        k: int = 10
    ) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Measures ranking quality with logarithmic discount
        
        Args:
            recommendations: List of recommended item IDs
            actual_items: List of actual relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG score (0-1)
        """
        if not recommendations or not actual_items or k <= 0:
            return 0.0
        
        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k]):
            if rec in actual_items:
                # Gain is 1 (binary relevance), discounted by log position
                dcg += 1.0 / np.log2(i + 2)  # +2 because index starts at 0
        
        # IDCG: Ideal DCG (perfect ranking - all relevant items first)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(
        self, 
        recommendations: List[str], 
        actual_items: List[str], 
        k: int = 5
    ) -> float:
        """
        Hit Rate@K: Binary indicator if at least one relevant item is recommended
        
        Args:
            recommendations: List of recommended item IDs
            actual_items: List of actual relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        if not recommendations or not actual_items or k <= 0:
            return 0.0
        
        top_k_recs = set(recommendations[:k])
        actual_set = set(actual_items)
        
        return 1.0 if len(top_k_recs.intersection(actual_set)) > 0 else 0.0
    
    def coverage(
        self, 
        all_recommendations: List[List[str]], 
        total_items: int
    ) -> float:
        """
        Catalog Coverage: Proportion of items that were ever recommended
        Measures how well the system explores the item catalog
        
        Args:
            all_recommendations: List of recommendation lists
            total_items: Total number of items in catalog
            
        Returns:
            Coverage ratio (0-1)
        """
        if total_items <= 0:
            return 0.0
        
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / total_items
    
    def diversity(self, recommendations: List[Dict[str, Any]]) -> float:
        """
        Recommendation Diversity: Measures variety in recommendations
        Combines location diversity and price dispersion
        
        Args:
            recommendations: List of recommendation dictionaries with metadata
            
        Returns:
            Diversity score (0-1)
        """
        if len(recommendations) <= 1:
            return 0.0
        
        # Extract features
        locations = [rec.get('location', 'Unknown') for rec in recommendations]
        prices = [rec.get('avg_price', 0) for rec in recommendations]
        
        # Remove invalid prices
        valid_prices = [p for p in prices if p > 0]
        
        # Location diversity (entropy-based)
        unique_locations = len(set(locations))
        location_diversity = unique_locations / len(recommendations)
        
        # Price diversity (coefficient of variation, capped at 1)
        if len(valid_prices) > 1:
            mean_price = np.mean(valid_prices)
            std_price = np.std(valid_prices)
            price_diversity = min(std_price / mean_price, 1.0) if mean_price > 0 else 0.0
        else:
            price_diversity = 0.0
        
        # Combined diversity (equal weight)
        return (location_diversity + price_diversity) / 2
    
    def personalization(
        self, 
        all_recommendations: List[List[str]]
    ) -> float:
        """
        Personalization: Measures how different recommendations are across users
        Higher score = more personalized recommendations
        
        Args:
            all_recommendations: List of recommendation lists for different users
            
        Returns:
            Personalization score (0-1)
        """
        if len(all_recommendations) <= 1:
            return 0.0
        
        # Calculate pairwise Jaccard distance
        n_users = len(all_recommendations)
        total_distance = 0.0
        comparisons = 0
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                set_i = set(all_recommendations[i])
                set_j = set(all_recommendations[j])
                
                # Jaccard distance = 1 - Jaccard similarity
                if len(set_i.union(set_j)) > 0:
                    jaccard_sim = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                    total_distance += (1 - jaccard_sim)
                    comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def evaluate_model(
        self, 
        rec_engine, 
        k_values: List[int] = [3, 5, 10], 
        sample_users: Optional[int] = 50,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation across multiple metrics
        
        Args:
            rec_engine: Recommendation engine with hybrid_recommendations method
            k_values: List of K values to evaluate
            sample_users: Number of users to sample (None = all users)
            verbose: Print progress information
            
        Returns:
            Dictionary of metric scores
        """
        if self.test_interactions is None:
            raise ValueError("Must run create_train_test_split() first!")
        
        if verbose:
            print(f"\n{'='*70}")
            print("EVALUATING RECOMMENDATION MODEL")
            print(f"{'='*70}")
        
        # Get unique test users
        test_users = self.test_interactions['userCode'].unique()
        
        # Sample users if requested
        if sample_users and len(test_users) > sample_users:
            np.random.seed(42)
            test_users = np.random.choice(test_users, sample_users, replace=False)
            if verbose:
                print(f"Evaluating on {sample_users} sampled users")
        else:
            if verbose:
                print(f"Evaluating on {len(test_users)} users")
        
        results = defaultdict(list)
        all_recommendations = []
        all_recommendation_details = []
        successful_evals = 0
        failed_evals = 0
        
        for idx, user in enumerate(test_users):
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{len(test_users)} users evaluated")
            
            # Get actual test items for this user
            actual_items = self.test_interactions[
                self.test_interactions['userCode'] == user
            ]['hotel_name'].tolist()
            
            if not actual_items:
                continue
            
            # Get recommendations
            try:
                recs = rec_engine.hybrid_recommendations(
                    user, 
                    n_recommendations=max(k_values),
                    debug=False
                )
                
                if not recs:
                    failed_evals += 1
                    continue
                
                rec_hotels = [rec['hotel_name'] for rec in recs]
                all_recommendations.append(rec_hotels)
                all_recommendation_details.append(recs)
                
                # Calculate metrics for each k
                for k in k_values:
                    results[f'precision@{k}'].append(
                        self.precision_at_k(rec_hotels, actual_items, k)
                    )
                    results[f'recall@{k}'].append(
                        self.recall_at_k(rec_hotels, actual_items, k)
                    )
                    results[f'f1@{k}'].append(
                        self.f1_score_at_k(rec_hotels, actual_items, k)
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
                
                successful_evals += 1
                
            except Exception as e:
                if verbose:
                    warnings.warn(f"Error evaluating user {user}: {e}")
                failed_evals += 1
                continue
        
        if successful_evals == 0:
            raise ValueError("No successful evaluations. Check your recommendation engine.")
        
        if verbose:
            print(f"\nSuccessful evaluations: {successful_evals}")
            print(f"Failed evaluations: {failed_evals}")
        
        # Calculate averages
        avg_results = {
            metric: np.mean(values) 
            for metric, values in results.items()
        }
        
        # Calculate coverage
        total_hotels = self.user_hotel_matrix.shape[1]
        avg_results['coverage'] = self.coverage(all_recommendations, total_hotels)
        
        # Calculate personalization
        avg_results['personalization'] = self.personalization(all_recommendations)
        
        # Add metadata
        avg_results['_metadata'] = {
            'n_users_evaluated': successful_evals,
            'n_users_failed': failed_evals,
            'total_items': total_hotels,
            'k_values': k_values
        }
        
        return avg_results
    
    def print_evaluation_results(self, results: Dict[str, float]):
        """
        Pretty print evaluation results with interpretations
        
        Args:
            results: Dictionary of evaluation metrics
        """
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Accuracy Metrics
        print("\n ACCURACY METRICS:")
        print("-" * 70)
        
        metric_groups = {
            'Precision': [k for k in results.keys() if k.startswith('precision@')],
            'Recall': [k for k in results.keys() if k.startswith('recall@')],
            'F1-Score': [k for k in results.keys() if k.startswith('f1@')],
            'NDCG': [k for k in results.keys() if k.startswith('ndcg@')],
            'Hit Rate': [k for k in results.keys() if k.startswith('hit_rate@')]
        }
        
        for group_name, metrics in metric_groups.items():
            if metrics:
                print(f"\n  {group_name}:")
                for metric in sorted(metrics):
                    print(f"    {metric.upper():<20}: {results[metric]:.4f}")
        
        if 'map' in results:
            print(f"\n  Mean Average Precision:")
            print(f"    {'MAP':<20}: {results['map']:.4f}")
        
        # Business Metrics
        print("\n BUSINESS METRICS:")
        print("-" * 70)
        
        if 'coverage' in results:
            coverage_pct = results['coverage'] * 100
            print(f"  {'Catalog Coverage':<25}: {results['coverage']:.4f} ({coverage_pct:.2f}%)")
        
        if 'diversity' in results:
            print(f"  {'Recommendation Diversity':<25}: {results['diversity']:.4f}")
        
        if 'personalization' in results:
            print(f"  {'Personalization':<25}: {results['personalization']:.4f}")
        
        # Interpretations
        print("\n INTERPRETATION:")
        print("-" * 70)
        
        # Precision interpretation
        if 'precision@5' in results:
            p5 = results['precision@5']
            if p5 > 0.3:
                status = "EXCELLENT"
                desc = "Recommendations are highly relevant"
            elif p5 > 0.2:
                status = "GOOD"
                desc = "Recommendations are relevant"
            elif p5 > 0.1:
                status = "FAIR"
                desc = "Room for improvement in relevance"
            else:
                status = "POOR"
                desc = "Need better recommendation algorithms"
            print(f"  Precision@5 ({p5:.2%}) - {status}: {desc}")
        
        # NDCG interpretation
        if 'ndcg@10' in results:
            ndcg = results['ndcg@10']
            if ndcg > 0.5:
                print(f"  NDCG@10 ({ndcg:.2%}) - GOOD: Ranking quality is strong")
            elif ndcg > 0.3:
                print(f"  NDCG@10 ({ndcg:.2%}) - FAIR: Ranking could be improved")
            else:
                print(f"  NDCG@10 ({ndcg:.2%}) - POOR: Ranking needs work")
        
        # Coverage interpretation
        if 'coverage' in results:
            cov = results['coverage']
            if cov > 0.5:
                print(f"  Coverage ({cov:.2%}) - GOOD: Broad catalog exposure")
            elif cov > 0.3:
                print(f"  Coverage ({cov:.2%}) - FAIR: Moderate catalog exposure")
            else:
                print(f"  Coverage ({cov:.2%}) - LOW: Too focused on few items")
        
        # Diversity interpretation
        if 'diversity' in results:
            div = results['diversity']
            if div > 0.6:
                print(f"  Diversity ({div:.2f}) - EXCELLENT: Highly varied recommendations")
            elif div > 0.4:
                print(f"  Diversity ({div:.2f}) - GOOD: Reasonably varied")
            else:
                print(f"  Diversity ({div:.2f}) - LOW: Recommendations too similar")
        
        # Personalization interpretation
        if 'personalization' in results:
            pers = results['personalization']
            if pers > 0.7:
                print(f"  Personalization ({pers:.2f}) - EXCELLENT: Highly personalized")
            elif pers > 0.5:
                print(f"  Personalization ({pers:.2f}) - GOOD: Reasonably personalized")
            else:
                print(f"  Personalization ({pers:.2f}) - LOW: Too generic")
    
    def plot_evaluation_results(
        self, 
        results: Dict[str, float],
        save_path: str = '../outputs/evaluation_metrics.png'
    ):
        """
        Visualize evaluation metrics
        
        Args:
            results: Dictionary of evaluation metrics
            save_path: Path to save the plot
        """
        # Set style
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Recommendation System Evaluation', fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Precision/Recall/F1/NDCG at different K
        k_metrics = {}
        for metric, value in results.items():
            if '@' in metric and not metric.startswith('_'):
                k = int(metric.split('@')[1])
                metric_name = metric.split('@')[0]
                if k not in k_metrics:
                    k_metrics[k] = {}
                k_metrics[k][metric_name] = value
        
        if k_metrics:
            k_values = sorted(k_metrics.keys())
            precision_vals = [k_metrics[k].get('precision', 0) for k in k_values]
            recall_vals = [k_metrics[k].get('recall', 0) for k in k_values]
            f1_vals = [k_metrics[k].get('f1', 0) for k in k_values]
            ndcg_vals = [k_metrics[k].get('ndcg', 0) for k in k_values]
            
            axes[0, 0].plot(k_values, precision_vals, marker='o', label='Precision', linewidth=2.5, markersize=8)
            axes[0, 0].plot(k_values, recall_vals, marker='s', label='Recall', linewidth=2.5, markersize=8)
            axes[0, 0].plot(k_values, f1_vals, marker='^', label='F1-Score', linewidth=2.5, markersize=8)
            axes[0, 0].plot(k_values, ndcg_vals, marker='d', label='NDCG', linewidth=2.5, markersize=8)
            axes[0, 0].set_xlabel('K (Number of Recommendations)', fontsize=11)
            axes[0, 0].set_ylabel('Score', fontsize=11)
            axes[0, 0].set_title('Accuracy Metrics vs K', fontsize=13, fontweight='bold')
            axes[0, 0].legend(loc='best', frameon=True)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, max(max(precision_vals + recall_vals + f1_vals + ndcg_vals), 0.1) * 1.1)
        
        # Plot 2: Hit Rate
        if k_metrics:
            hit_rates = [k_metrics[k].get('hit_rate', 0) for k in k_values]
            bars = axes[0, 1].bar(k_values, hit_rates, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)
            axes[0, 1].set_xlabel('K', fontsize=11)
            axes[0, 1].set_ylabel('Hit Rate', fontsize=11)
            axes[0, 1].set_title('Hit Rate@K', fontsize=13, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            axes[0, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.2f}',
                              ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Business Metrics
        business_metrics = []
        business_values = []
        colors = []
        
        metric_config = [
            ('coverage', 'Coverage', '#2ecc71'),
            ('diversity', 'Diversity', '#3498db'),
            ('personalization', 'Personalization', '#9b59b6'),
            ('map', 'MAP', '#e74c3c')
        ]
        
        for key, label, color in metric_config:
            if key in results:
                business_metrics.append(label)
                business_values.append(results[key])
                colors.append(color)
        
        if business_metrics:
            bars = axes[1, 0].bar(business_metrics, business_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            axes[1, 0].set_ylabel('Score', fontsize=11)
            axes[1, 0].set_title('Business Metrics', fontsize=13, fontweight='bold')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].tick_params(axis='x', rotation=15)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}',
                              ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 4: Metrics Comparison Heatmap
        metric_summary = {}
        for k in [3, 5, 10]:
            for metric_type in ['precision', 'recall', 'f1', 'ndcg']:
                key = f'{metric_type}@{k}'
                if key in results:
                    if metric_type not in metric_summary:
                        metric_summary[metric_type] = {}
                    metric_summary[metric_type][f'@{k}'] = results[key]
        
        if metric_summary:
            df_heatmap = pd.DataFrame(metric_summary).T
            sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[1, 1], cbar_kws={'label': 'Score'},
                       linewidths=1, linecolor='white')
            axes[1, 1].set_title('Metrics Heatmap', fontsize=13, fontweight='bold')
            axes[1, 1].set_xlabel('K Value', fontsize=11)
            axes[1, 1].set_ylabel('Metric Type', fontsize=11)
        
        plt.tight_layout()
        
        # Save plot
        try:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Evaluation plot saved to '{save_path}'")
        except Exception as e:
            print(f"\n  Could not save plot: {e}")
        
        plt.show()
    
    def compare_models(
        self,
        models: Dict[str, Any],
        k_values: List[int] = [5, 10],
        sample_users: Optional[int] = 50
    ) -> pd.DataFrame:
        """
        Compare multiple recommendation models
        
        Args:
            models: Dictionary of {model_name: model_instance}
            k_values: List of K values to evaluate
            sample_users: Number of users to sample
            
        Returns:
            DataFrame with comparison results
        """
        print(f"\n{'='*70}")
        print("COMPARING MULTIPLE MODELS")
        print(f"{'='*70}")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\n Evaluating: {model_name}")
            results = self.evaluate_model(
                model,
                k_values=k_values,
                sample_users=sample_users,
                verbose=False
            )
            comparison_results[model_name] = results
        
        # Create comparison DataFrame
        metrics_to_compare = [
            'precision@5', 'recall@5', 'ndcg@5', 'hit_rate@5',
            'precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10',
            'map', 'coverage', 'diversity', 'personalization'
        ]
        
        comparison_df = pd.DataFrame({
            model_name: {
                metric: results.get(metric, 0.0)
                for metric in metrics_to_compare
            }
            for model_name, results in comparison_results.items()
        }).T
        
        # Add ranking
        for metric in comparison_df.columns:
            comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print("\n", comparison_df[metrics_to_compare].to_string())
        
        return comparison_df
    
    def statistical_significance_test(
        self,
        model_a_results: List[float],
        model_b_results: List[float],
        test: str = 't-test'
    ) -> Dict[str, float]:
        """
        Test statistical significance between two models
        
        Args:
            model_a_results: List of metric scores for model A
            model_b_results: List of metric scores for model B
            test: Type of test ('t-test' or 'wilcoxon')
            
        Returns:
            Dictionary with test statistic and p-value
        """
        from scipy import stats
        
        if test == 't-test':
            statistic, p_value = stats.ttest_ind(model_a_results, model_b_results)
        elif test == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(model_a_results, model_b_results)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }


# ============================================================================
# USAGE FUNCTIONS
# ============================================================================

def run_evaluation(
    rec_engine,
    complete_df: pd.DataFrame,
    user_hotel_matrix: pd.DataFrame,
    test_size: float = 0.2,
    min_interactions: int = 3,
    k_values: List[int] = [3, 5, 10],
    sample_users: Optional[int] = 50,
    save_results: bool = True
) -> Tuple[Dict[str, float], 'RecommendationEvaluator']:
    """
    Complete evaluation pipeline with all steps
    
    Args:
        rec_engine: Recommendation engine instance
        complete_df: Complete interaction data
        user_hotel_matrix: User-hotel matrix
        test_size: Test set proportion
        min_interactions: Minimum interactions per user
        k_values: List of K values to evaluate
        sample_users: Number of users to sample (None = all)
        save_results: Whether to save results to files
        
    Returns:
        Tuple of (results_dict, evaluator_instance)
    """
    print(f"\n{'='*70}")
    print("RUNNING PROFESSIONAL EVALUATION PIPELINE")
    print(f"{'='*70}")
    
    # Initialize evaluator
    print("\n[1/5] Initializing evaluator...")
    evaluator = RecommendationEvaluator(complete_df, user_hotel_matrix)
    
    # Create train/test split
    print("\n[2/5] Creating train/test split...")
    train_data, test_data = evaluator.create_train_test_split(
        test_size=test_size,
        min_interactions=min_interactions
    )
    
    # Evaluate model
    print("\n[3/5] Evaluating model...")
    results = evaluator.evaluate_model(
        rec_engine,
        k_values=k_values,
        sample_users=sample_users,
        verbose=True
    )
    
    # Print results
    print("\n[4/5] Generating report...")
    evaluator.print_evaluation_results(results)
    
    # Visualize results
    print("\n[5/5] Creating visualizations...")
    evaluator.plot_evaluation_results(results)
    
    # Save results if requested
    if save_results:
        try:
            import json
            import os
            
            # Create output directory
            os.makedirs('../outputs', exist_ok=True)
            
            # Save JSON results (remove metadata for JSON)
            results_to_save = {k: v for k, v in results.items() if not k.startswith('_')}
            with open('../outputs/evaluation_results.json', 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            # Save CSV results
            results_df = pd.DataFrame([results_to_save])
            results_df.to_csv('../outputs/evaluation_results.csv', index=False)
            
            # Save train/test split
            train_data.to_csv('../outputs/train_interactions.csv', index=False)
            test_data.to_csv('../outputs/test_interactions.csv', index=False)
            
            print("\n✓ Results saved to:")
            print("  - ../outputs/evaluation_results.json")
            print("  - ../outputs/evaluation_results.csv")
            print("  - ../outputs/train_interactions.csv")
            print("  - ../outputs/test_interactions.csv")
            print("  - ../outputs/evaluation_metrics.png")
            
        except Exception as e:
            print(f"\n  Could not save results: {e}")
    
    print(f"\n{'='*70}")
    print("✓ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    
    return results, evaluator


def quick_evaluate(
    rec_engine,
    complete_df: pd.DataFrame,
    user_hotel_matrix: pd.DataFrame,
    k: int = 5
) -> Dict[str, float]:
    """
    Quick evaluation with minimal output
    
    Args:
        rec_engine: Recommendation engine
        complete_df: Complete interaction data
        user_hotel_matrix: User-hotel matrix
        k: K value for metrics
        
    Returns:
        Dictionary of key metrics
    """
    evaluator = RecommendationEvaluator(complete_df, user_hotel_matrix)
    evaluator.create_train_test_split(test_size=0.2, min_interactions=3)
    
    results = evaluator.evaluate_model(
        rec_engine,
        k_values=[k],
        sample_users=30,
        verbose=False
    )
    
    # Return only key metrics
    key_metrics = {
        f'precision@{k}': results.get(f'precision@{k}', 0),
        f'recall@{k}': results.get(f'recall@{k}', 0),
        f'ndcg@{k}': results.get(f'ndcg@{k}', 0),
        'coverage': results.get('coverage', 0),
        'diversity': results.get('diversity', 0)
    }
    
    print("\nQuick Evaluation Results:")
    for metric, value in key_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return key_metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("RECOMMENDATION EVALUATOR - TEST MODE")
    print("="*80)
    
    # Create mock data for demonstration
    print("\n1. Creating mock data...")
    
    np.random.seed(42)
    
    # Mock user-hotel matrix
    n_users = 50
    n_hotels = 30
    user_hotel_matrix = pd.DataFrame(
        np.random.poisson(2, (n_users, n_hotels)),
        index=[f'U{i}' for i in range(n_users)],
        columns=[f'Hotel_{chr(65+i)}' for i in range(n_hotels)]
    )
    
    # Mock complete_df
    complete_data = []
    for user_idx, user in enumerate(user_hotel_matrix.index):
        for hotel_idx, hotel in enumerate(user_hotel_matrix.columns):
            rating = user_hotel_matrix.loc[user, hotel]
            if rating > 0:
                complete_data.append({
                    'userCode': user,
                    'hotel_name': hotel,
                    'rating': rating
                })
    
    complete_df = pd.DataFrame(complete_data)
    
    print(f"   ✓ Created {len(user_hotel_matrix)} users")
    print(f"   ✓ Created {user_hotel_matrix.shape[1]} hotels")
    print(f"   ✓ Created {len(complete_df)} interactions")
    
    # Create mock recommendation engine
    print("\n2. Creating mock recommendation engine...")
    
    class MockRecommendationEngine:
        def __init__(self, user_hotel_matrix):
            self.matrix = user_hotel_matrix
        
        def hybrid_recommendations(self, user_code, n_recommendations=10, debug=False):
            """Mock recommendations - returns random hotels"""
            all_hotels = self.matrix.columns.tolist()
            np.random.shuffle(all_hotels)
            
            recommendations = []
            for i, hotel in enumerate(all_hotels[:n_recommendations]):
                recommendations.append({
                    'hotel_name': hotel,
                    'location': np.random.choice(['NYC', 'LA', 'SF']),
                    'avg_price': np.random.uniform(100, 300),
                    'avg_stay': np.random.uniform(2, 5),
                    'recommendation_score': 1.0 - (i * 0.1)
                })
            
            return recommendations
    
    mock_engine = MockRecommendationEngine(user_hotel_matrix)
    print("   ✓ Mock engine created")
    
    # Run evaluation
    print("\n3. Running evaluation pipeline...")
    
    try:
        results, evaluator = run_evaluation(
            rec_engine=mock_engine,
            complete_df=complete_df,
            user_hotel_matrix=user_hotel_matrix,
            test_size=0.2,
            min_interactions=3,
            k_values=[3, 5, 10],
            sample_users=20,
            save_results=False  # Don't save in test mode
        )
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nThe evaluation module is ready for production use!")
        print("\nTo use with your actual recommendation engine:")
        print("""
from evaluation_module import run_evaluation

results, evaluator = run_evaluation(
    rec_engine=your_recommendation_engine,
    complete_df=your_complete_data,
    user_hotel_matrix=your_user_hotel_matrix,
    k_values=[3, 5, 10],
    sample_users=50,
    save_results=True
)
        """)
        
    except Exception as e:
        print(f"\n Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# INTEGRATION EXAMPLE FOR YOUR PROJECT
# ============================================================================

"""
COMPLETE INTEGRATION EXAMPLE:

# After loading your data and creating your recommendation engine:

from evaluation_module import run_evaluation, quick_evaluate

# Option 1: Full evaluation with all metrics
print("\\n" + "="*70)
print("RUNNING COMPREHENSIVE EVALUATION")
print("="*70)

results, evaluator = run_evaluation(
    rec_engine=rec_engine,
    complete_df=complete_df,
    user_hotel_matrix=user_hotel_matrix,
    test_size=0.2,
    min_interactions=3,
    k_values=[3, 5, 10],
    sample_users=50,
    save_results=True
)

# Option 2: Quick evaluation for rapid testing
quick_results = quick_evaluate(
    rec_engine=rec_engine,
    complete_df=complete_df,
    user_hotel_matrix=user_hotel_matrix,
    k=5
)

# Option 3: Compare multiple models
from evaluation_module import RecommendationEvaluator

evaluator = RecommendationEvaluator(complete_df, user_hotel_matrix)
evaluator.create_train_test_split(test_size=0.2, min_interactions=3)

models_to_compare = {
    'Hybrid Model': hybrid_model,
    'Content-Based': content_based_model,
    'Collaborative': collaborative_model
}

comparison_df = evaluator.compare_models(
    models=models_to_compare,
    k_values=[5, 10],
    sample_users=50
)

print("\\nModel Comparison:")
print(comparison_df)
"""
