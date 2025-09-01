# Dashboard de Métriques PPO Flappy Bird 📊

Ce système de dashboard génère des visualisations complètes pour analyser l'entraînement de votre agent PPO sur Flappy Bird. Parfait pour vos présentations et analyses !

## 🚀 Fonctionnalités

- **Dashboard Complet** : 8 graphiques différents avec métriques détaillées
- **Courbes d'Entraînement** : Récompenses, losses, convergence
- **Métriques Avancées** : Efficacité d'apprentissage, stabilité, tendances
- **Export pour Présentations** : Graphiques haute résolution pour diapositives
- **Intégration Automatique** : Se génère pendant l'entraînement
- **Script Autonome** : Génération à partir de données existantes

## 📈 Types de Graphiques Générés

### 1. **Récompenses d'Épisodes**
- Récompenses brutes avec moyenne mobile
- Ligne de tendance pour voir l'amélioration
- Analyse de la progression

### 2. **Losses d'Entraînement** 
- Actor Loss et Critic Loss
- Évolution de la stabilité d'entraînement

### 3. **Métriques de Performance**
- Tendance des récompenses (slope)
- Efficacité d'apprentissage
- Indicateur de convergence

### 4. **Évolution de la Politique**
- Entropie de la politique
- KL Divergence
- Changements dans la stratégie

### 5. **Analyse d'Efficacité**
- Efficacité d'apprentissage dans le temps
- Indicateur de convergence

### 6. **Analyse de Convergence**
- Distribution des returns
- Variance et stabilité

### 7. **Résumé Statistique**
- Métriques clés actuelles
- Vue d'ensemble textuelle

### 8. **Heatmap de Performance**
- Performance récente visualisée
- Patterns dans les derniers épisodes

## 🔧 Installation et Configuration

### Dépendances
```bash
pip install matplotlib seaborn pandas numpy torch tensorboard
```

### Structure des Fichiers
```
flappy_bird_ppo/
├── metrics_dashboard.py      # Système principal de dashboard
├── generate_dashboard.py     # Script autonome
├── main.py                  # Entraînement (modifié avec dashboard)
├── metrics/                 # Données de métriques (auto-généré)
└── dashboard_exports/       # Exports de graphiques (auto-généré)
```

## 📊 Utilisation

### 1. **Pendant l'Entraînement (Automatique)**
Le dashboard se génère automatiquement quand vous lancez l'entraînement :

```bash
python main.py --mode train
```

- Métriques collectées toutes les 10 épisodes
- Dashboard généré à chaque SAVE_FREQUENCY
- Dashboard final à la fin de l'entraînement

### 2. **Génération Manuelle à partir de Métriques**
```bash
python generate_dashboard.py --mode metrics --input_dir metrics --output_dir dashboard_exports
```

### 3. **Génération à partir de Logs TensorBoard**
```bash
python generate_dashboard.py --mode tensorboard --input_dir logs --output_dir dashboard_exports
```

### 4. **Génération de Slides de Présentation**
```bash
python generate_dashboard.py --mode slides --output_dir presentation_slides
```

### 5. **Utilisation Programmatique**
```python
from metrics_dashboard import MetricsDashboard, create_dashboard_from_agent

# Pendant l'entraînement
dashboard_path = create_dashboard_from_agent(
    agent=agent,
    config=config,
    episode=current_episode,
    training_stats=training_stats,
    evaluation_stats=eval_stats
)

# Ou création manuelle
dashboard = MetricsDashboard()
dashboard_path = dashboard.create_comprehensive_dashboard()
```

## 🎨 Options de Personnalisation

### Configuration des Graphiques
Dans `metrics_dashboard.py`, vous pouvez modifier :
- Couleurs des graphiques
- Taille des fenêtres de moyenne mobile
- Métriques affichées
- Style des graphiques

### Fréquence de Génération
Dans `main.py`, modifiez la condition :
```python
# Générer dashboard tous les N épisodes
if episode % N == 0:
    dashboard.create_comprehensive_dashboard()
```

## 📁 Fichiers Générés

### Dashboard Principal
- `ppo_dashboard_YYYYMMDD_HHMMSS.png` : Dashboard complet (20x16 inches, 300 DPI)

### Graphiques Individuels
- `rewards_YYYYMMDD_HHMMSS.png` : Graphique des récompenses
- `losses_YYYYMMDD_HHMMSS.png` : Graphique des losses

### Slides de Présentation
- `slide_1_overview_YYYYMMDD_HHMMSS.png` : Vue d'ensemble de l'entraînement
- `slide_2_performance_YYYYMMDD_HHMMSS.png` : Analyse de performance
- `slide_3_summary_YYYYMMDD_HHMMSS.png` : Résumé des métriques clés

### Données
- `metrics/metrics_episode_N.json` : Métriques détaillées par épisode
- `metrics_summary_YYYYMMDD_HHMMSS.txt` : Résumé textuel

## 🎯 Pour vos Présentations

### Recommandations
1. **Utilisez les slides générées** : Format optimisé pour présentations
2. **Dashboard complet** : Pour analyses détaillées
3. **Graphiques individuels** : Pour focus sur aspects spécifiques

### Qualité d'Image
- Résolution : 300 DPI (qualité impression)
- Format : PNG avec fond blanc
- Taille : Optimisée pour projections

### Contenus Suggérés pour Diapositives

#### Slide 1 : Introduction
- "Entraînement PPO sur Flappy Bird"
- Graphique des récompenses avec tendance

#### Slide 2 : Analyse Technique
- Losses d'entraînement
- Métriques de convergence

#### Slide 3 : Résultats
- Dashboard complet ou résumé statistique
- Performance finale

## 🐛 Dépannage

### Problèmes Courants

**Erreur : "No training history found"**
```bash
# Vérifiez que l'entraînement a été lancé et que des métriques existent
ls metrics/
```

**Graphiques vides ou erreurs matplotlib**
```bash
# Vérifiez les dépendances
pip install --upgrade matplotlib seaborn
```

**Erreur de mémoire**
```bash
# Réduisez la résolution dans metrics_dashboard.py
plt.savefig(path, dpi=150)  # au lieu de 300
```

### Personnalisation Avancée

Pour modifier les métriques collectées, éditez `_compute_derived_metrics()` dans `metrics_dashboard.py`.

Pour changer l'apparence, modifiez les styles au début de `metrics_dashboard.py` :
```python
plt.style.use('your_preferred_style')
sns.set_palette("your_palette")
```

## 📊 Exemples de Métriques

Le système génère automatiquement ces métriques avancées :
- **Reward Trend Slope** : Pente de l'amélioration
- **Learning Efficiency** : Vitesse d'apprentissage
- **Convergence Indicator** : Stabilité de la performance  
- **Policy Change Rate** : Vitesse de changement de politique
- **Value Estimation Accuracy** : Précision du critique

## 🎉 Résultat Final

Vous obtiendrez :
- ✅ Dashboard complet avec 8 graphiques professionnels
- ✅ Slides prêtes pour présentation
- ✅ Métriques détaillées exportées
- ✅ Graphiques individuels haute résolution
- ✅ Analyse statistique complète

Parfait pour vos présentations, rapports, et analyses d'entraînement ! 🚀