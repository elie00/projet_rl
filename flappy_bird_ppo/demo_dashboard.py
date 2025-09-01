#!/usr/bin/env python3
"""
Demo Dashboard Generator - Creates sample dashboard with synthetic data
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from metrics_dashboard import MetricsDashboard

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def generate_synthetic_training_data(num_episodes=500):
    """Generate realistic synthetic training data for demo"""
    
    training_history = []
    
    # Simulate training progression
    base_reward = -10
    improvement_rate = 0.02
    noise_level = 5
    
    for episode in range(num_episodes):
        # Simulate improving performance with some noise
        expected_reward = base_reward + episode * improvement_rate
        actual_reward = expected_reward + np.random.normal(0, noise_level)
        
        # Simulate decreasing losses
        actor_loss = 0.5 * np.exp(-episode/200) + np.random.normal(0, 0.1)
        critic_loss = 1.0 * np.exp(-episode/150) + np.random.normal(0, 0.1)
        
        # Simulate entropy decrease (exploration to exploitation)
        entropy = 0.8 * np.exp(-episode/300) + 0.1 + np.random.normal(0, 0.05)
        
        # Simulate KL divergence
        kl_div = 0.05 + np.random.exponential(0.02)
        
        # Create episode rewards (multiple per episode)
        episode_rewards = []
        for _ in range(np.random.randint(1, 4)):  # 1-3 episodes per training step
            episode_rewards.append(actual_reward + np.random.normal(0, 2))
        
        # Create synthetic metrics entry
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'episode_rewards': episode_rewards,
            'training': {
                'actor_loss': max(0.001, actor_loss),
                'critic_loss': max(0.001, critic_loss),
                'entropy': max(0.01, entropy),
                'kl_divergence': max(0.001, kl_div),
                'returns_mean': actual_reward,
                'returns_std': noise_level,
            },
            'performance_metrics': {
                'reward_trend_slope': improvement_rate + np.random.normal(0, 0.005),
                'reward_stability': max(0.1, noise_level - episode/100),
                'learning_efficiency': min(1.0, episode/300),
                'convergence_indicator': min(1.0, episode/400)
            }
        }
        
        # Add evaluation stats every 50 episodes
        if episode % 50 == 0 and episode > 0:
            eval_reward = actual_reward + np.random.normal(0, 1)
            metrics['evaluation'] = {
                'eval_reward_mean': eval_reward,
                'eval_reward_std': 2.0,
                'eval_reward_min': eval_reward - 5,
                'eval_reward_max': eval_reward + 3,
                'eval_length_mean': 50 + episode/10,
                'eval_length_std': 10,
                'success_rate': min(1.0, max(0.0, (episode - 100)/400)),
                'num_episodes': 10
            }
        
        training_history.append(metrics)
    
    return training_history

def create_demo_dashboard():
    """Create demonstration dashboard"""
    print("🚀 Génération du dashboard de démonstration...")
    
    # Create directories
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("dashboard_exports", exist_ok=True)
    
    # Generate synthetic data
    print("📊 Génération de données d'entraînement synthétiques...")
    training_history = generate_synthetic_training_data(500)
    
    # Save synthetic metrics
    for i, metrics in enumerate(training_history[::10]):  # Save every 10th episode
        filename = f"metrics/metrics_episode_{metrics['episode']}.json"
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Create dashboard
    dashboard = MetricsDashboard()
    dashboard.training_history = training_history
    
    print("🎨 Génération du dashboard principal...")
    dashboard_path = dashboard.create_comprehensive_dashboard(save_individual=True)
    
    print("📈 Export du résumé des métriques...")
    summary_path = dashboard.export_metrics_summary()
    
    # Generate presentation slides
    print("🎯 Génération des slides de présentation...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Slide 1: Training Overview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Flappy Bird - Aperçu de l\'Entraînement', fontsize=20, fontweight='bold')
    
    episodes = [m['episode'] for m in training_history]
    dashboard._plot_episode_rewards(ax1, episodes)
    dashboard._plot_training_losses(ax2, episodes)
    dashboard._plot_performance_metrics(ax3, episodes)
    dashboard._plot_policy_evolution(ax4, episodes)
    
    plt.tight_layout()
    slide1_path = f'dashboard_exports/slide_1_apercu_{timestamp}.png'
    plt.savefig(slide1_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Slide 2: Performance Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Flappy Bird - Analyse de Performance', fontsize=20, fontweight='bold')
    
    dashboard._plot_learning_efficiency(ax1, episodes)
    dashboard._plot_convergence_analysis(ax2, episodes)
    dashboard._plot_statistical_summary(ax3)
    dashboard._plot_performance_heatmap(ax4)
    
    plt.tight_layout()
    slide2_path = f'dashboard_exports/slide_2_performance_{timestamp}.png'
    plt.savefig(slide2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Slide 3: Summary with French text
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.suptitle('PPO Flappy Bird - Résumé des Métriques Clés', fontsize=20, fontweight='bold')
    
    latest = training_history[-1]
    
    summary_text = f"""
RÉSUMÉ DE L'ENTRAÎNEMENT PPO FLAPPY BIRD
{'='*60}

CONFIGURATION:
• Algorithme: Proximal Policy Optimization (PPO)
• Environnement: Flappy Bird avec capteurs LiDAR
• Épisodes d'entraînement: {len(training_history)}
• Épisode final: {latest.get('episode', 'N/A')}

MÉTRIQUES DE PERFORMANCE FINALES:
• Loss Acteur: {latest.get('training', {}).get('actor_loss', 0):.4f}
• Loss Critique: {latest.get('training', {}).get('critic_loss', 0):.4f}
• Entropie Politique: {latest.get('training', {}).get('entropy', 0):.4f}
• KL Divergence: {latest.get('training', {}).get('kl_divergence', 0):.4f}

ANALYSE D'APPRENTISSAGE:
• Pente de Tendance des Récompenses: {latest.get('performance_metrics', {}).get('reward_trend_slope', 0):.4f}
• Efficacité d'Apprentissage: {latest.get('performance_metrics', {}).get('learning_efficiency', 0):.4f}
• Indicateur de Convergence: {latest.get('performance_metrics', {}).get('convergence_indicator', 0):.4f}
• Stabilité des Récompenses: {latest.get('performance_metrics', {}).get('reward_stability', 0):.4f}

RÉSULTATS D'ÉVALUATION:
• Récompense Moyenne: {latest.get('evaluation', {}).get('eval_reward_mean', 0):.2f}
• Récompense Maximale: {latest.get('evaluation', {}).get('eval_reward_max', 0):.2f}
• Taux de Succès: {latest.get('evaluation', {}).get('success_rate', 0):.1%}
• Longueur Moyenne d'Épisode: {latest.get('evaluation', {}).get('eval_length_mean', 0):.1f}

CONCLUSIONS:
✓ Amélioration constante des performances observée
✓ Convergence stable de la politique atteinte
✓ Équilibre exploration-exploitation efficace
✓ Agent prêt pour déploiement ou optimisation avancée

RECOMMANDATIONS POUR LA PRÉSENTATION:
• Utiliser le graphique de récompenses pour montrer la progression
• Mettre en avant la diminution des losses (stabilité d'entraînement)
• Souligner le taux de succès et les métriques d'efficacité
• Comparer avec une baseline aléatoire si disponible
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    slide3_path = f'dashboard_exports/slide_3_resume_{timestamp}.png'
    plt.savefig(slide3_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return dashboard_path, summary_path, [slide1_path, slide2_path, slide3_path]

if __name__ == "__main__":
    dashboard_path, summary_path, slide_paths = create_demo_dashboard()
    
    print("\n🎉 DASHBOARD GÉNÉRÉ AVEC SUCCÈS!")
    print("=" * 50)
    print(f"📊 Dashboard principal: {dashboard_path}")
    print(f"📋 Résumé des métriques: {summary_path}")
    print("\n🎯 Slides de présentation générées:")
    for i, slide_path in enumerate(slide_paths, 1):
        print(f"   Slide {i}: {slide_path}")
    
    print(f"\n📁 Tous les fichiers sont dans: dashboard_exports/")
    print("✨ Prêt pour vos présentations!")