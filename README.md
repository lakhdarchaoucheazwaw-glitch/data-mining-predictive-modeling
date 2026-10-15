# Fouille de Données & Modélisation Prédictive (Hiver 2025)

Ce projet aborde deux problématiques classiques de la modélisation statistique et décisionnelle :
1. **Classification supervisée & Extraction de règles métier :** Construction d'un arbre de décision interprétable pour prédire l'intention d'achat client et en déduire des règles stratégiques.
2. **Analyse de séries temporelles stochastiques :** Modélisation probabiliste du cours du pétrole brut via un mouvement brownien géométrique, et estimation empirique de l'espérance, de la volatilité et de l'autocovariance.

---

## 1. Arbres de Décision & Aide à la Décision
Pour éviter l'effet "boîte noire" et permettre la prise de décision managériale, le modèle est volontairement contraint en profondeur (`max_depth=3`). 
- **Critère de découpe :** Gain d'information basé sur l'entropie de Shannon :
  $$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$
- **Extraction :** Transformation du graphe de décision en règles logiques explicites (`IF-THEN`).

---

## 2. Séries Temporelles du Pétrole & Probabilités
Le prix brut $P_t$ d'une matière première n'étant pas stationnaire, l'analyse porte sur les rendements logarithmiques 

Le script estime :
- **L'espérance empirique** $\mathbb{E}[R]$ (tendance journalière et annualisée).
- **La volatilité annualisée**
- **L'autocovariance** empirique pour mesurer la mémoire du marché à différents retards 

---

