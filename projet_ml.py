"""
Auteur : Azouaou Lakhdarchaouche
Description : Implémentation de deux modèles prédictifs et statistiques pour l'aide à la décision :
    1. Arbre de décision régularisé pour l'extraction de règles symboliques (comportement d'achat).
    2. Modélisation stochastique (GBM) et analyse probabiliste de séries temporelles (pétrole brut).
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score


def generate_customer_data(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Génère un échantillon synthétique et réaliste de données clients.
    
    Les variables explicatives suivent des distributions statistiques standards 
    (Normale, Exponentielle, Poisson) et la variable cible est générée via
    une fonction de score latent non-linéaire avec bruit gaussien.

    Paramètres :
        n_samples (int) : Taille de l'échantillon à générer.
        seed (int) : Graine pour le générateur pseudo-aléatoire (reproductibilité).

    Retourne :
        pd.DataFrame : Tableau de données contenant les attributs et le label cible.
    """
    # Utilisation du générateur moderne NumPy (recommandé pour la recherche)
    rng = np.random.default_rng(seed)
    
    # 1. Caractéristiques socio-démographiques et comportementales
    age = rng.integers(18, 70, size=n_samples)
    # Revenu modélisé par une loi normale tronquée entre 15k et 100k EUR
    income = np.round(rng.normal(45.0, 15.0, size=n_samples).clip(15.0, 100.0), 1)
    # Temps de visite modélisé par une loi exponentielle (comportement web typique)
    dwell_time = np.round(rng.exponential(scale=8.0, size=n_samples) + 1.0, 1)
    # Nombre d'articles dans le panier modélisé par un processus de comptage (Poisson)
    cart_items = rng.poisson(lam=2.5, size=n_samples)
    # Score CSAT (Customer Satisfaction) discret de 1 à 5
    satisfaction = rng.integers(1, 6, size=n_samples)

    # 2. Modèle latent : relation non-linéaire avec interactions métier explicites
    # P(Achat) augmente si : (longue visite ET panier non vide) OU (haute satisfaction)
    latent_score = (
        0.10 
        + 0.50 * ((dwell_time > 10.0) & (cart_items >= 2)) 
        + 0.30 * (satisfaction >= 4)
        + rng.normal(0.0, 0.05, n_samples)  # Aléa résiduel / bruit irréductible
    )
    # Binarisation selon un seuil de décision à 0.45
    target = (latent_score > 0.45).astype(int)

    return pd.DataFrame({
        'Age': age,
        'Revenu_kEUR': income,
        'Temps_Site_Min': dwell_time,
        'Nb_Articles': cart_items,
        'Satisfaction': satisfaction,
        'Achat': target
    })


def train_decision_tree(
    df: pd.DataFrame, 
    features: List[str], 
    target: str, 
    max_depth: int = 3
) -> Tuple[DecisionTreeClassifier, float]:
    """
    Entraîne un classifieur par arbre de décision avec contraintes de régularisation.
    
    L'arbre est volontairement contraint en profondeur (max_depth=3) pour servir de
    modèle en 'boîte blanche' (white-box), garantissant l'extraction de règles
    logiques interprétables pour la prise de décision stratégique.

    Paramètres :
        df (pd.DataFrame) : Jeu de données source.
        features (List[str]) : Liste des variables explicatives.
        target (str) : Nom de la variable binaire à prédire.
        max_depth (int) : Profondeur maximale de l'arbre (évite le surapprentissage).

    Retourne :
        Tuple[DecisionTreeClassifier, float] : Modèle ajusté et exactitude (accuracy).
    """
    X = df[features]
    y = df[target]

    # Découpage 75/25 avec stratification pour préserver le ratio des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Utilisation de l'Entropie de Shannon H(S) pour maximiser le gain d'information
    # min_samples_leaf=20 assure une significativité statistique minimale par règle
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Évaluation de la capacité de généralisation sur l'échantillon test
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("=== RAPPORT DE CLASSIFICATION (ARBRE DE DÉCISION) ===")
    print(classification_report(y_test, y_pred, target_names=["Non-Acheteur", "Acheteur"]))
    
    return clf, acc


def simulate_geometric_brownian_motion(
    s0: float = 75.0, 
    mu: float = 0.05, 
    sigma: float = 0.35, 
    n_days: int = 1000, 
    seed: int = 42
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Simule la trajectoire de prix d'un actif via un Mouvement Brownien Géométrique (GBM).
    
    Équation Différentielle Stochastique (EDS) sous-jacente :
        dS_t = mu * S_t * dt + sigma * S_t * dW_t
    Résolue analytiquement par le lemme d'Itô :
        S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)

    Paramètres :
        s0 (float) : Prix initial du baril en USD.
        mu (float) : Dérive (drift) théorique annualisée.
        sigma (float) : Volatilité théorique annualisée (35% typique pour le brut).
        n_days (int) : Nombre d'observations journalières simulées.
        seed (int) : Graine du générateur stochastique.

    Retourne :
        Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray] : Dates, série des prix et log-rendements.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0  # Fréquence quotidienne (252 jours ouvrés par an)

    # Création du calendrier de cotation (hors week-ends)
    dates = pd.date_range(start="2021-01-01", periods=n_days, freq='B')
    
    # Simulation des accroissements browniens gaussiens dW_t ~ N(0, dt)
    dz = rng.normal(0.0, np.sqrt(dt), size=n_days)
    
    # Calcul exact des rendements logarithmiques journaliers : r_t = ln(S_t / S_{t-1})
    log_returns = (mu - 0.5 * (sigma ** 2)) * dt + sigma * dz
    
    # Intégration stochastique : trajectoire cumulée exponentielle
    prices = s0 * np.exp(np.insert(np.cumsum(log_returns), 0, 0.0))[:-1]

    return dates, prices, log_returns


def compute_time_series_statistics(
    log_returns: np.ndarray, 
    max_lag: int = 10
) -> Tuple[float, float, float, np.ndarray]:
    """
    Calcule les estimateurs non biaisés de l'espérance, de la variance et de l'autocovariance.
    
    La fonction d'autocovariance empirique mesure la dépendance temporelle :
        gamma(k) = Cov(R_t, R_{t-k}) = (1/N) * sum((R_t - E[R]) * (R_{t-k} - E[R]))

    Paramètres :
        log_returns (np.ndarray) : Vecteur des rendements continus (stationnaires).
        max_lag (int) : Horizon temporel maximal (en jours) pour l'autocovariance.

    Retourne :
        Tuple : E[R] journalière, E[R] annualisée, volatilité annualisée, vecteur d'autocovariance.
    """
    # 1. Espérance mathématique empirique
    e_daily = float(np.mean(log_returns))
    e_annual = e_daily * 252.0  # Linéarité de l'espérance sur 252 jours

    # 2. Volatilité (écart-type) annualisée corrigée (facteur sqrt(252))
    sigma_annual = float(np.std(log_returns, ddof=1) * np.sqrt(252.0))

    # 3. Calcul vectorisé de l'autocovariance aux différents retards (lags k)
    n = len(log_returns)
    centered = log_returns - e_daily
    autocov = np.array([
        np.sum(centered[:n - k] * centered[k:]) / n if k > 0 else np.var(log_returns, ddof=0)
        for k in range(max_lag + 1)
    ])

    return e_daily, e_annual, sigma_annual, autocov


def main() -> None:
    """Point d'entrée du programme : exécute la modélisation et affiche les résultats."""
    
    # -------------------------------------------------------------------------
    # 1. Modélisation Supervisée : Arbre de décision & Extraction de règles
    # -------------------------------------------------------------------------
    features = ['Age', 'Revenu_kEUR', 'Temps_Site_Min', 'Nb_Articles', 'Satisfaction']
    df_customers = generate_customer_data(n_samples=1200)

    model, test_accuracy = train_decision_tree(df_customers, features, target='Achat')
    print(f"Précision globale (Accuracy) : {test_accuracy:.2%}\n")

    # Extraction des règles logiques au format symbolique lisible (Aide à la décision)
    print("=== RÈGLES DE DÉCISION EXTRAITES ===")
    print(export_text(model, feature_names=features))

    # -------------------------------------------------------------------------
    # 2. Modélisation Stochastique : Espérance et Structure de Covariance
    # -------------------------------------------------------------------------
    dates, prices, returns = simulate_geometric_brownian_motion()
    e_daily, e_annual, sigma_annual, autocov = compute_time_series_statistics(returns, max_lag=10)

    print("=== ESTIMATIONS PROBABILISTES (PÉTROLE BRUT) ===")
    print(f"Espérance journalière E[R] : {e_daily:.6f}")
    print(f"Espérance annualisée       : {e_annual:.2%}")
    print(f"Volatilité annualisée (σ)  : {sigma_annual:.2%}\n")

    print("Vecteur d'autocovariance gamma(k) pour k in [0, 5] :")
    for k in range(6):
        print(f"  Lag {k} : {autocov[k]:.8f}")

    # -------------------------------------------------------------------------
    # 3. Visualisations Analytiques
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10))

    # Graphe 1 : Structure arborescente (Interprétabilité des règles)
    ax1 = plt.subplot(2, 2, (1, 2))
    plot_tree(model, feature_names=features, class_names=["Non-Achat", "Achat"], 
              filled=True, rounded=True, ax=ax1, fontsize=9)
    ax1.set_title("Arbre de Décision Décisionnel (Segmentation & Règles)")

    # Graphe 2 : Série temporelle non-stationnaire et espérance locale glissante
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(dates, prices, color='black', alpha=0.7, label='Prix WTI ($/baril)')
    # La moyenne mobile sur 50 jours sert d'estimateur local de la tendance
    ax2.plot(dates, pd.Series(prices, index=dates).rolling(50).mean(), 
             color='red', label='Espérance locale (50j)')
    ax2.set_title("Dynamique du Prix Brut")
    ax2.set_ylabel("USD")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Graphe 3 : Décroissance de l'autocovariance (absence de mémoire = efficience)
    ax3 = plt.subplot(2, 2, 4)
    ax3.stem(range(len(autocov)), autocov)
    ax3.set_title("Fonction d'Autocovariance des Rendements")
    ax3.set_xlabel("Décalage k (jours)")
    ax3.set_ylabel("Covariance")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
