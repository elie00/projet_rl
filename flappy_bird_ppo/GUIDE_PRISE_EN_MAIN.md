# 🚀 Guide de Prise en Main - Flappy Bird PPO

Ce guide vous accompagne pas à pas pour comprendre, installer et utiliser ce projet d'apprentissage par renforcement avec l'algorithme PPO.

## 📋 Table des Matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Prérequis et installation](#prérequis-et-installation)
3. [Première utilisation](#première-utilisation)
4. [Comprendre la structure](#comprendre-la-structure)
5. [Guide des commandes](#guide-des-commandes)
6. [Interpréter les résultats](#interpréter-les-résultats)
7. [Personnaliser l'entraînement](#personnaliser-lentraînement)
8. [Résolution de problèmes](#résolution-de-problèmes)
9. [Cas d'usage avancés](#cas-dusage-avancés)

## 🎯 Vue d'ensemble du projet

### Qu'est-ce que ce projet fait ?

Ce projet entraîne un agent intelligent à jouer au jeu Flappy Bird en utilisant l'**apprentissage par renforcement**. L'agent apprend par essais et erreurs, sans instructions préalables, jusqu'à maîtriser parfaitement le jeu.

### Pourquoi utiliser PPO ?

**PPO (Proximal Policy Optimization)** est un algorithme de pointe qui :
- ✅ Apprend efficacement avec peu d'échantillons
- ✅ Reste stable pendant l'entraînement
- ✅ Fonctionne bien sur de nombreux environnements
- ✅ Est utilisé dans l'industrie (OpenAI, DeepMind)

### Résultats attendus

Après entraînement, votre agent devrait :
- Obtenir **30-80 pipes en moyenne** (vs -7 pour une politique aléatoire)
- Atteindre un **taux de succès de 100%**
- Naviguer avec **fluidité et précision** dans le jeu

## 💻 Prérequis et installation

### Système requis

- **Python 3.8+** (recommandé : Python 3.10)
- **4 GB RAM minimum** (8 GB recommandé)
- **2 GB d'espace disque**
- **MacOS, Linux ou Windows**

### Installation étape par étape

1. **Vérifiez votre version Python :**
   ```bash
   python3.10 --version
   # Devrait afficher : Python 3.10.x
   ```

2. **Naviguez dans le répertoire du projet :**
   ```bash
   cd flappy_bird_ppo
   ```

3. **Installez les dépendances :**
   ```bash
   # Installation des dépendances de base
   pip3.10 install -r requirements.txt
   
   # Installation de TensorFlow compatible
   pip3.10 install tensorflow==2.16.1
   
   # Installation des outils vidéo
   pip3.10 install "gymnasium[other]" moviepy
   ```

4. **Testez l'installation :**
   ```bash
   python3.10 -c "import torch, gymnasium, flappy_bird_gymnasium; print('✅ Installation réussie !')"
   ```

### Résolution d'erreurs d'installation

**Erreur : "command not found: python3.10"**
```bash
# Sur macOS avec Homebrew
brew install python@3.10

# Sur Ubuntu/Debian
sudo apt install python3.10 python3.10-pip

# Sur Windows, téléchargez depuis python.org
```

**Erreur : "No module named 'torch'"**
```bash
pip3.10 install --upgrade torch torchvision
```

## 🏃‍♂️ Première utilisation

### Test rapide (5 minutes)

1. **Lancement d'un entraînement court :**
   ```bash
   python3.10 main.py --mode train
   ```
   Vous devriez voir :
   ```
   Starting PPO training for Flappy Bird
   Device: cpu
   Total episodes: 10000
   Environment: FlappyBird-v0
   ```

2. **Laissez tourner 2-3 minutes**, puis arrêtez avec `Ctrl+C`

3. **Évaluez l'agent :**
   ```bash
   python3.10 main.py --mode eval --model_path models/best_model.pt --episodes 5
   ```

### Que se passe-t-il pendant l'entraînement ?

1. **Collecte d'expérience** : L'agent joue 512 étapes
2. **Calcul des récompenses** : Évalue les bonnes/mauvaises actions
3. **Mise à jour des réseaux** : Améliore la politique de jeu
4. **Évaluation périodique** : Teste les performances tous les 100 épisodes
5. **Sauvegarde automatique** : Garde le meilleur modèle

## 📁 Comprendre la structure

```
flappy_bird_ppo/
├── 📄 main.py              # Point d'entrée principal
├── ⚙️ config.py            # Tous les paramètres
├── 🧠 models.py            # Réseaux Actor et Critic
├── 🤖 ppo_agent.py         # Logique d'apprentissage PPO
├── 🛠️ utils.py             # Fonctions utilitaires
├── 📋 requirements.txt     # Liste des dépendances
├── 📖 README.md            # Documentation complète
├── 📖 GUIDE_PRISE_EN_MAIN.md # Ce guide
├── 📁 videos/              # Vidéos d'évaluation
├── 📁 logs/                # Logs TensorBoard
└── 📁 models/              # Modèles sauvegardés
```

### Fichiers clés à connaître

**`config.py`** - Tous les paramètres :
- Hyperparamètres PPO
- Paramètres d'entraînement
- Configuration du logging

**`main.py`** - Interface principale :
- Mode entraînement
- Mode évaluation
- Mode visualisation

**`models/best_model.pt`** - Meilleur modèle sauvegardé

## 🎮 Guide des commandes

### Commandes essentielles

**1. Entraînement complet :**
```bash
python3.10 main.py --mode train
```
- Durée : 2-4 heures
- Crée : logs, vidéos, modèles
- Sortie : Rapport final avec statistiques

**2. Évaluation d'un modèle :**
```bash
python3.10 main.py --mode eval --model_path models/best_model.pt --episodes 10
```
- Tests le modèle sur 10 épisodes
- Affiche statistiques détaillées
- Compare avec baseline aléatoire

**3. Regarder l'agent jouer :**
```bash
python3.10 main.py --mode watch --model_path models/best_model.pt --episodes 3
```
- Ouvre une fenêtre de jeu
- Montre l'agent en action
- Parfait pour les démonstrations

### Commandes avancées

**Reprendre un entraînement :**
```bash
python3.10 main.py --mode train --resume models/ppo_flappy_bird_1000.pt
```

**Évaluation avec rendu :**
```bash
python3.10 main.py --mode eval --model_path models/best_model.pt --episodes 20 --render
```

**Test rapide :**
```bash
python3.10 main.py --mode eval --model_path models/best_model.pt --episodes 3
```

## 📊 Interpréter les résultats

### Pendant l'entraînement

**Sortie console typique :**
```
Episode 500/10000
  Avg Reward (last 100): 13.57
  Max Reward (last 100): 27.20
  Actor Loss: -0.0012
  Critic Loss: 1.5072
  Time per episode: 0.16s
  Total steps: 256512
```

**Comment lire ces métriques :**
- **Avg Reward** : Score moyen récent (plus c'est haut, mieux c'est)
- **Max Reward** : Meilleur score récent
- **Actor Loss** : Perte de la politique (devrait diminuer)
- **Critic Loss** : Perte de l'évaluation d'état
- **Time per episode** : Vitesse d'entraînement

### Après évaluation

**Rapport d'évaluation :**
```
TRAINED AGENT PERFORMANCE:
  Average Reward: 35.10 ± 20.32
  Best Reward: 66.90
  Success Rate: 100.0%

PERFORMANCE COMPARISON:
  Reward Improvement: +42.55 (+571.4%)
  ✓ Agent outperforms random baseline!
```

**Critères de succès :**
- **Reward moyen > 10** : Agent compétent
- **Success Rate > 80%** : Performance consistante
- **Amélioration > 300%** : Apprentissage significatif

### Visualiser avec TensorBoard

```bash
tensorboard --logdir logs/
```
Puis ouvrez http://localhost:6006 pour voir :
- Courbes d'apprentissage
- Évolution des losses
- Statistiques d'entraînement

## ⚙️ Personnaliser l'entraînement

### Modifier les hyperparamètres

Éditez `config.py` :

```python
# Pour un entraînement plus rapide (moins précis)
MAX_STEPS_PER_ROLLOUT = 256    # Au lieu de 512
PPO_EPOCHS = 2                 # Au lieu de 4

# Pour plus d'exploration
ENTROPY_COEFF = 0.05           # Au lieu de 0.01

# Pour un entraînement plus long
TOTAL_EPISODES = 20000         # Au lieu de 10000
```

### Configurations prédéfinies

**Configuration "Rapide" (test) :**
```python
TOTAL_EPISODES = 1000
MAX_STEPS_PER_ROLLOUT = 256
EVAL_FREQUENCY = 50
```

**Configuration "Performance" (production) :**
```python
TOTAL_EPISODES = 20000
MAX_STEPS_PER_ROLLOUT = 1024
PPO_EPOCHS = 8
```

**Configuration "Demo" (présentation) :**
```python
RECORD_VIDEO = True
EVAL_FREQUENCY = 50
RENDER_EVAL = False
```

### Désactiver fonctionnalités

**Pas de vidéos (plus rapide) :**
```python
RECORD_VIDEO = False
```

**Pas de Weights & Biases :**
```python
USE_WANDB = False
```

## 🔧 Résolution de problèmes

### Erreurs courantes

**1. "ModuleNotFoundError: No module named 'torch'"**
```bash
pip3.10 install torch torchvision
```

**2. "CUDA out of memory"**
Dans `config.py` :
```python
DEVICE = "cpu"  # Force l'utilisation du CPU
MAX_STEPS_PER_ROLLOUT = 256  # Réduit la mémoire
```

**3. "Render mode is None"**
Erreur corrigée automatiquement dans le code.

**4. L'agent ne s'améliore pas**
- Vérifiez que `LEARNING_RATE = 3e-4` (pas trop haut/bas)
- Augmentez `TOTAL_EPISODES`
- Vérifiez les rewards dans les logs

### Performance lente

**Sur CPU :**
- Réduisez `MAX_STEPS_PER_ROLLOUT` à 256
- Désactivez l'enregistrement vidéo
- Utilisez moins d'épisodes pour les tests

**Manque d'espace disque :**
- Vidéos prennent ~500 MB pour un entraînement complet
- Supprimez les anciennes vidéos dans `videos/`
- Désactivez `RECORD_VIDEO = False`

### Debugging avancé

**Mode verbose :**
```python
# Dans main.py, ajoutez au début :
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Vérifier les gradients :**
```python
# Dans config.py :
MAX_GRAD_NORM = 0.5  # Clipping plus strict
```

## 🎯 Cas d'usage avancés

### 1. Comparaison d'hyperparamètres

Créez plusieurs configs et comparez :

```bash
# Config 1 : LR élevé
python3.10 main.py --mode train  # Modifiez config.py avec LR=1e-3

# Config 2 : LR faible  
python3.10 main.py --mode train  # Modifiez config.py avec LR=1e-5

# Comparez les résultats dans logs/
```

### 2. Transfert d'apprentissage

```bash
# Entraînez sur un environnement
python3.10 main.py --mode train

# Reprenez sur un environnement modifié
python3.10 main.py --mode train --resume models/best_model.pt
```

### 3. Ensemble de modèles

Entraînez plusieurs modèles avec des graines différentes :
```python
# config.py
SEED = 42  # Premier modèle
SEED = 123 # Deuxième modèle  
SEED = 456 # Troisième modèle
```

### 4. Analyse de robustesse

Testez sur différents environnements :
```bash
# Testez sur plusieurs épisodes
python3.10 main.py --mode eval --model_path models/best_model.pt --episodes 100

# Analysez les statistiques de variance
```

## 🎓 Conseils pédagogiques

### Pour un cours/présentation

1. **Démonstration baseline** (2 min) :
   - Montrez une politique aléatoire qui échoue
   - Score typique : -7 à 0 pipes

2. **Lancement entraînement** (5 min) :
   - Expliquez PPO rapidement
   - Montrez les métriques en temps réel

3. **Agent entraîné** (3 min) :
   - Démonstration avec `--mode watch`
   - Scores typiques : 30-80 pipes

### Pour un projet étudiant

**Expérimentations possibles :**
- Comparer PPO vs politique aléatoire
- Impact des hyperparamètres
- Courbes d'apprentissage
- Analyse de convergence

**Livrables suggérés :**
- Code fonctionnel ✅
- Vidéos d'évaluation 📹
- Rapport avec métriques 📊
- Analyse des résultats 📈

## 📚 Ressources supplémentaires

### Documentation technique
- [Article PPO original](https://arxiv.org/abs/1707.06347)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [PyTorch RL tutorials](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

### Environnements similaires
- CartPole-v1 : Plus simple, pour débuter
- Atari Breakout : Plus complexe, plus long
- Lunar Lander : Problème de contrôle continu

### Extensions possibles
- **Autres algorithmes** : A3C, SAC, TD3
- **Autres jeux** : Super Mario Bros, Pac-Man
- **Réseaux avancés** : CNN pour images, LSTM pour mémoire

---

## 🎉 Félicitations !

Si vous êtes arrivé jusqu'ici, vous maîtrisez maintenant :
- ✅ L'installation et utilisation du projet
- ✅ L'interprétation des résultats
- ✅ La personnalisation des paramètres
- ✅ Le débogage des problèmes courants

Votre agent PPO devrait obtenir des scores impressionnants (30-80 pipes) et vous avez toutes les clés pour impressionner votre audience ! 🚀

**Bon apprentissage par renforcement !** 🤖