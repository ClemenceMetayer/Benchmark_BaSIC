# Benchmark des hyperparamètres BaSIC

Benchmark systématique des hyperparamètres de la méthode BaSIC (Bayesian Sparse Identification of Coupled dynamical systems) sur 5 systèmes ODE : Lotka-Volterra, Chain, SEIR, Goldbeter et Yeast Glycolysis.

Le plan expérimental suit une approche **OFAT** (One-Factor-At-a-Time) autour de deux baselines (horseshoe et spike-and-slab), en variant chaque hyperparamètre indépendamment.

## Structure du projet

```
├── config/                          # Configuration du plan d'expériences
│   ├── generate_plan.py             # Génère le plan OFAT (~105 configs)
│   ├── experiment_plan.csv          # Plan généré
│   └── hyperparameter_methods.csv   # Référence des hyperparamètres et leur rôle
│
├── metrics/                         # Calcul des métriques d'évaluation
│   ├── benchmark_metrics.py         # F1, NMSPE, NMSE, CI coverage, diagnostics MCMC
│   └── patch_log_likelihood.py      # Ajout post-hoc de la log-vraisemblance
│
├── visualization/                   # Scripts de visualisation
│   ├── plot_results.py              # Heatmaps, sensibilité, PCA, timing, comparaisons
│   ├── plot_trajectories.py         # Trajectoires des 5 systèmes ODE
│   └── analyze_recommendations.py   # Recommandations par hyperparamètre
│
├── figures/                         # Figures générées
├── run_all_experiments.py           # Point d'entrée principal
├── generate_fake_results.py         # Génère des résultats synthétiques pour le dev
├── all_benchmark_results.csv        # Résultats agrégés de toutes les expériences
├── .gitignore
└── README.md
```

## Utilisation

### 1. Générer le plan d'expériences

```bash
python config/generate_plan.py
```

Produit `config/experiment_plan.csv` contenant toutes les configurations à tester.

### 2. Lancer les expériences

```bash
# Toutes les expériences (séquentiel)
python run_all_experiments.py

# Filtrer par système et/ou run
python run_all_experiments.py --systems lotka_volterra chain --runs 0 1 2

# Exécution parallèle
python run_all_experiments.py --workers 4

# Sans refit ni métriques
python run_all_experiments.py --skip-refit --skip-metrics

# Plan d'expériences personnalisé
python run_all_experiments.py --plan my_plan.csv
```

Les résultats sont sauvés dans des dossiers `out_<système>_run<id>/` et agrégés dans `all_benchmark_results.csv`.

Les expériences déjà complétées sont automatiquement ignorées (reprise possible).

### 3. Patch post-hoc (log-vraisemblance)

```bash
python metrics/patch_log_likelihood.py
```

### 4. Visualiser les résultats

```bash
# Figures d'analyse du benchmark (heatmaps, sensibilité, PCA, timing, etc.)
python visualization/plot_results.py

# Trajectoires des systèmes ODE
python visualization/plot_trajectories.py

# Recommandations par hyperparamètre (score composite, delta vs baseline)
python visualization/analyze_recommendations.py
```

### Développement (résultats synthétiques)

```bash
# Générer des résultats factices pour tester les scripts de visualisation
python generate_fake_results.py
```

## Métriques évaluées

| Catégorie | Métrique | Description |
|-----------|----------|-------------|
| Structure | F1-score, Hamming | Récupération de la structure (termes actifs vs vrais) |
| Paramètres | NMSPE | Erreur normalisée sur les coefficients estimés |
| Prédiction | NMSE training, NMSE new x0 | Qualité de prédiction sur données d'entraînement et nouvelle CI |
| Couverture | CI coverage | Proportion des vrais paramètres dans les intervalles de crédibilité |
| Log-vraisemblance | Médiane, moyenne | Adéquation du modèle aux observations |
| Diagnostics | R-hat, ESS | Convergence MCMC |
| Temps | Sparse, refit, total | Temps de calcul |

## Hyperparamètres testés

- **Prior** : tau0, slab_scale, degree_penalty (horseshoe) ; spike_sd, slab_sd, theta_a/b (spike-and-slab)
- **MCMC** : warmup, samples, chains, thinning, target_accept, max_treedepth
- **Sélection** : CI level (horseshoe), inclusion probability threshold (spike-and-slab)
- **Librairie** : degré polynomial, biais, termes Michaelis-Menten
- **Intégrateur** : single/multiple shooting, segments, tolérances, max_steps
- **Initialisation** : stratégie (regression/median), bruit
- **Données** : fraction de bruit, réplicats, horizon temporel

Voir `config/hyperparameter_methods.csv` pour la liste complète avec valeurs par défaut et localisation dans le code.
# Benchmark_BaSIC
