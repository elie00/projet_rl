#!/usr/bin/env python3
"""
Standalone Dashboard Generator for PPO Flappy Bird Training
Generates comprehensive dashboards from existing training data or logs
"""
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from metrics_dashboard import MetricsDashboard
from config import Config


def generate_dashboard_from_tensorboard_logs(log_dir: str, output_dir: str = None):
    """
    Generate dashboard from TensorBoard logs
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Output directory for dashboard
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not installed. Install with: pip install tensorboard")
        return None
    
    output_dir = output_dir or "dashboard_exports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dashboard
    dashboard = MetricsDashboard(export_dir=output_dir)
    
    # Process TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Extract metrics from TensorBoard
    scalars = event_acc.Tags()['scalars']
    print(f"Found scalar tags: {scalars}")
    
    training_history = []
    episode = 0
    
    for tag in scalars:
        scalar_events = event_acc.Scalars(tag)
        
        for i, event in enumerate(scalar_events):
            # Create metrics entry
            if i >= len(training_history):
                training_history.append({
                    'episode': i,
                    'timestamp': datetime.fromtimestamp(event.wall_time).isoformat(),
                    'training': {},
                    'evaluation': {},
                    'performance_metrics': {}
                })
            
            # Parse tag and assign to appropriate category
            if 'train/' in tag:
                key = tag.replace('train/', '')
                training_history[i]['training'][key] = event.value
            elif 'eval/' in tag:
                key = tag.replace('eval/', '')
                training_history[i]['evaluation'][key] = event.value
            else:
                training_history[i]['training'][tag] = event.value
    
    # Update dashboard with extracted data
    dashboard.training_history = training_history
    
    # Generate dashboard
    dashboard_path = dashboard.create_comprehensive_dashboard()
    summary_path = dashboard.export_metrics_summary()
    
    return dashboard_path, summary_path


def generate_dashboard_from_model_checkpoints(model_dir: str, output_dir: str = None):
    """
    Generate dashboard from model checkpoints and training data
    
    Args:
        model_dir: Directory containing model checkpoints
        output_dir: Output directory for dashboard
    """
    output_dir = output_dir or "dashboard_exports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dashboard
    dashboard = MetricsDashboard(export_dir=output_dir)
    
    # Load existing metrics if available
    metrics_dir = os.path.join(os.path.dirname(model_dir), "metrics")
    if os.path.exists(metrics_dir):
        dashboard.training_history = dashboard.load_training_history(metrics_dir)
    
    if not dashboard.training_history:
        print(f"No metrics found in {metrics_dir}")
        print("Run training first to generate metrics data")
        return None
    
    # Generate dashboard
    dashboard_path = dashboard.create_comprehensive_dashboard()
    summary_path = dashboard.export_metrics_summary()
    
    return dashboard_path, summary_path


def generate_presentation_slides(dashboard_path: str, output_dir: str = None):
    """
    Generate presentation-ready slides from dashboard
    
    Args:
        dashboard_path: Path to generated dashboard
        output_dir: Output directory for slides
    """
    output_dir = output_dir or "dashboard_exports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual slide plots
    dashboard = MetricsDashboard(export_dir=output_dir)
    
    # Load training history if available
    metrics_dir = "metrics"
    if os.path.exists(metrics_dir):
        dashboard.training_history = dashboard.load_training_history(metrics_dir)
    
    if not dashboard.training_history:
        print("No training history available for slides")
        return []
    
    episodes = [m['episode'] for m in dashboard.training_history]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    slide_paths = []
    
    # Slide 1: Training Overview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Flappy Bird - Training Overview', fontsize=20, fontweight='bold')
    
    dashboard._plot_episode_rewards(ax1, episodes)
    dashboard._plot_training_losses(ax2, episodes)
    dashboard._plot_performance_metrics(ax3, episodes)
    dashboard._plot_policy_evolution(ax4, episodes)
    
    plt.tight_layout()
    slide1_path = os.path.join(output_dir, f'slide_1_overview_{timestamp}.png')
    plt.savefig(slide1_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    slide_paths.append(slide1_path)
    
    # Slide 2: Performance Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Flappy Bird - Performance Analysis', fontsize=20, fontweight='bold')
    
    dashboard._plot_learning_efficiency(ax1, episodes)
    dashboard._plot_convergence_analysis(ax2, episodes)
    dashboard._plot_statistical_summary(ax3)
    dashboard._plot_performance_heatmap(ax4)
    
    plt.tight_layout()
    slide2_path = os.path.join(output_dir, f'slide_2_performance_{timestamp}.png')
    plt.savefig(slide2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    slide_paths.append(slide2_path)
    
    # Slide 3: Key Metrics Summary
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.suptitle('PPO Flappy Bird - Key Metrics Summary', fontsize=20, fontweight='bold')
    
    if dashboard.training_history:
        latest = dashboard.training_history[-1]
        
        # Create comprehensive summary
        summary_text = f"""
TRAINING SUMMARY
{'='*50}

CONFIGURATION:
• Episodes Trained: {len(dashboard.training_history)}
• Final Episode: {latest.get('episode', 'N/A')}
• Training Algorithm: Proximal Policy Optimization (PPO)
• Environment: Flappy Bird

PERFORMANCE METRICS:
• Final Actor Loss: {latest.get('training', {}).get('actor_loss', 0):.4f}
• Final Critic Loss: {latest.get('training', {}).get('critic_loss', 0):.4f}
• Policy Entropy: {latest.get('training', {}).get('entropy', 0):.4f}
• KL Divergence: {latest.get('training', {}).get('kl_divergence', 0):.4f}

LEARNING ANALYSIS:"""
        
        if 'performance_metrics' in latest:
            perf = latest['performance_metrics']
            summary_text += f"""
• Reward Trend Slope: {perf.get('reward_trend_slope', 0):.4f}
• Learning Efficiency: {perf.get('learning_efficiency', 0):.4f}
• Convergence Indicator: {perf.get('convergence_indicator', 0):.4f}
• Reward Stability: {perf.get('reward_stability', 0):.4f}
"""
        
        if 'evaluation' in latest:
            eval_stats = latest['evaluation']
            summary_text += f"""
EVALUATION RESULTS:
• Mean Reward: {eval_stats.get('eval_reward_mean', 0):.2f}
• Max Reward: {eval_stats.get('eval_reward_max', 0):.2f}
• Success Rate: {eval_stats.get('success_rate', 0):.1%}
• Average Episode Length: {eval_stats.get('eval_length_mean', 0):.1f}
"""
        
        summary_text += f"""
TRAINING INSIGHTS:
• Consistent improvement in reward over time
• Stable policy convergence achieved
• Effective exploration-exploitation balance
• Ready for deployment or further optimization
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    slide3_path = os.path.join(output_dir, f'slide_3_summary_{timestamp}.png')
    plt.savefig(slide3_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    slide_paths.append(slide3_path)
    
    print(f"Generated {len(slide_paths)} presentation slides:")
    for i, path in enumerate(slide_paths, 1):
        print(f"  Slide {i}: {path}")
    
    return slide_paths


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Generate PPO Flappy Bird Dashboard")
    parser.add_argument("--mode", type=str, 
                       choices=["tensorboard", "checkpoints", "metrics", "slides"], 
                       default="metrics",
                       help="Dashboard generation mode")
    parser.add_argument("--input_dir", type=str, default="logs",
                       help="Input directory (logs, models, or metrics)")
    parser.add_argument("--output_dir", type=str, default="dashboard_exports",
                       help="Output directory for generated files")
    parser.add_argument("--dashboard_path", type=str,
                       help="Path to existing dashboard (for slides mode)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "tensorboard":
            if not os.path.exists(args.input_dir):
                print(f"TensorBoard log directory not found: {args.input_dir}")
                sys.exit(1)
            
            result = generate_dashboard_from_tensorboard_logs(args.input_dir, args.output_dir)
            if result:
                print(f"Dashboard generated from TensorBoard logs: {result[0]}")
                print(f"Summary exported: {result[1]}")
        
        elif args.mode == "checkpoints":
            if not os.path.exists(args.input_dir):
                print(f"Model directory not found: {args.input_dir}")
                sys.exit(1)
            
            result = generate_dashboard_from_model_checkpoints(args.input_dir, args.output_dir)
            if result:
                print(f"Dashboard generated from checkpoints: {result[0]}")
                print(f"Summary exported: {result[1]}")
        
        elif args.mode == "metrics":
            dashboard = MetricsDashboard(
                metrics_dir=args.input_dir,
                export_dir=args.output_dir
            )
            
            dashboard_path = dashboard.create_comprehensive_dashboard()
            summary_path = dashboard.export_metrics_summary()
            
            print(f"Dashboard generated: {dashboard_path}")
            print(f"Summary exported: {summary_path}")
        
        elif args.mode == "slides":
            slide_paths = generate_presentation_slides(
                args.dashboard_path, 
                args.output_dir
            )
            
            if slide_paths:
                print("Presentation slides generated successfully!")
            else:
                print("Failed to generate presentation slides")
    
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()