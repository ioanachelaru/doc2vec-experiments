# Analiză comparativă: model.dv vs infer_vector — Django CVDP

## Context

Am rulat pipeline-ul de analiză cross-version pe **Django** (26 versiuni, 1.0–6.1, ~700–800 fișiere Python per versiune) cu două metode diferite de extragere a embedding-urilor Doc2Vec:

| | Run 1 — model.dv | Run 2 — infer_vector |
|---|---|---|
| **Metodă** | Embedding-uri deterministe din lookup table-ul intern | Embedding-uri calculate din token-urile documentului (200 epoci) |
| **GitHub Actions** | [Run 33478789208](https://github.com/ioanachelaru/doc2vec-experiments/actions/runs/33478789208) | [Run 33485534592](https://github.com/ioanachelaru/doc2vec-experiments/actions/runs/33485534592) |
| **Commit** | [`cad6162`](https://github.com/ioanachelaru/doc2vec-experiments/commit/cad6162) | [`4736121`](https://github.com/ioanachelaru/doc2vec-experiments/commit/4736121) |

Ambele rulări folosesc:
- Același model de bază (antrenat pe 100 repo-uri Python populare, [artifact](https://github.com/ioanachelaru/doc2vec-experiments/actions/runs/32718528477))
- Aceeași configurație: threshold=0.99, finetune epochs=10, vector_size=200
- Același set de date Django (subdirectorul `django/`, tag regex `^[0-9]+\.[0-9]+$`)
- Setup CVDP: versiunile 0..i = training, versiunea i+1 = test

## Problema descoperită cu model.dv

Antrenamentul cumulativ cauzează o problemă fundamentală cu `model.dv`: la fiecare reantrenare, gensim **realocă slot-urile** din vectorii de documente. Un fișier care avea slot-ul #42 în versiunea 1.0 poate primi slot-ul #42 alocat unui fișier complet diferit în versiunea 1.1.

**Rezultat:** fișiere fără legătură semantică primesc vectori identici (cosinus = 1.0), ceea ce generează sute de „coliziuni" false per pereche de versiuni.

Aceasta nu este o eroare de implementare — este o **limitare arhitecturală** a modului în care gensim gestionează `Doc2Vec.dv` în antrenamentul incremental. Vectorii din `model.dv` sunt optimizați în timpul antrenamentului ca un lookup table indexat numeric; când se adaugă documente noi, indexurile existente sunt redistribuite.

**Soluția:** `infer_vector()` calculează embedding-ul fiecărui document pornind de la token-urile sale reale, deci conținut identic → vector identic, indiferent de istoricul antrenamentului.

## Rezultate comparative

### Sumar

| Metrică | model.dv | infer_vector |
|---------|----------|--------------|
| **Leakage range** | 80–100% | 0.28–15.24% |
| **Perechi duplicate exacte** (sim=1.0) | 502–735 / pereche | 0 |
| **Coliziuni** (fișiere diferite, vectori similari) | 477–705 / pereche | 0–9 / pereche |
| **Same-file** (același fișier între versiuni) | 3–499 / pereche | 2–112 / pereche |
| **Timp de execuție** | 6 minute | 64 minute |

### Detaliu per pereche — model.dv

| Pereche | Train | Test | Test buggy | Test clean | Leaked buggy | Leaked clean | Same-file | Coliziuni |
|---------|-------|------|-----------|-----------|-------------|-------------|-----------|-----------|
| 1.0 → 1.1 | 502 | 545 | 148 | 489 | 130 (87.8%) | 372 (76.1%) | 25 | 477 |
| 1.1 → 1.2 | 1047 | 673 | 265 | 577 | 193 (72.8%) | 352 (61.0%) | 9 | 536 |
| 1.2 → 1.3 | 1720 | 728 | 335 | 572 | 306 (91.3%) | 367 (64.2%) | 17 | 664 |
| 1.3 → 1.4 | 2448 | 805 | 353 | 643 | 317 (89.8%) | 411 (63.9%) | 4 | 724 |
| 1.4 → 1.5 | 3253 | 807 | 312 | 689 | 312 (100%) | 493 (71.6%) | 49 | 801 |
| 1.5 → 1.6 | 4060 | 704 | 290 | 567 | 290 (100%) | 414 (73.0%) | 36 | 700 |
| 1.10 → 1.11 | 7623 | 712 | 293 | 557 | 293 (100%) | 419 (75.2%) | 314 | 711 |
| 1.11 → 2.0 | 8335 | 704 | 265 | 573 | 265 (100%) | 439 (76.6%) | 145 | 703 |
| 3.1 → 3.2 | 14067 | 716 | 177 | 681 | 177 (100%) | 539 (79.2%) | 183 | 715 |
| 5.1 → 5.2 | 17301 | 735 | 29 | 854 | 29 (100%) | 706 (82.7%) | 521 | 734 |

> **Notă:** De la versiunea 1.5, 100% din fișierele buggy apar ca leaked — un semnal clar de artefact.

### Detaliu per pereche — infer_vector

| Pereche | Train | Test | Test buggy | Test clean | Leaked buggy | Leaked clean | Same-file | Coliziuni |
|---------|-------|------|-----------|-----------|-------------|-------------|-----------|-----------|
| 1.0 → 1.1 | 502 | 545 | 148 | 489 | 8 (5.4%) | 61 (12.5%) | 67 | 2 |
| 1.1 → 1.2 | 1047 | 673 | 265 | 577 | 23 (8.7%) | 53 (9.2%) | 76 | 1 |
| 1.2 → 1.3 | 1720 | 728 | 335 | 572 | 40 (11.9%) | 51 (8.9%) | 91 | 0 |
| 1.3 → 1.4 | 2448 | 805 | 353 | 643 | 40 (11.3%) | 57 (8.9%) | 96 | 1 |
| 1.4 → 1.5 | 3253 | 807 | 312 | 689 | 6 (1.9%) | 42 (6.1%) | 48 | 1 |
| 1.5 → 1.6 | 4060 | 704 | 290 | 567 | 15 (5.2%) | 64 (11.3%) | 70 | 9 |
| 1.10 → 1.11 | 7623 | 712 | 293 | 557 | 36 (12.3%) | 29 (5.2%) | 65 | 0 |
| 1.11 → 2.0 | 8335 | 704 | 265 | 573 | 1 (0.4%) | 1 (0.2%) | 2 | 0 |
| 3.1 → 3.2 | 14067 | 716 | 177 | 681 | 40 (22.6%) | 37 (5.4%) | 77 | 0 |
| 4.1 → 4.2 | 15505 | 727 | 129 | 742 | 36 (27.9%) | 63 (8.5%) | 99 | 0 |
| 5.0 → 5.1 | 16583 | 731 | 80 | 799 | 24 (30.0%) | 81 (10.1%) | 105 | 0 |
| 5.1 → 5.2 | 17301 | 735 | 29 | 854 | 9 (31.0%) | 103 (12.1%) | 112 | 0 |

## Observații cheie

### 1. model.dv produce rezultate artificiale

Leakage-ul de 92–100% nu reflectă contaminarea reală a datelor, ci un **artefact al realocarilor de slot-uri** în antrenamentul cumulativ. Toate perechile duplicate aveau similaritate cosinus exactă = 1.0 — două fișiere diferite nu pot avea conținut identic la nivel de token, deci acesta este un semnal clar de artefact.

### 2. Leakage-ul real (infer_vector) este modest

**0.28–15.24%**, dominat de fișiere identice între versiuni consecutive (same_file). Aceasta este contaminare genuină — aceleași fișiere sursă existând neschimbate în versiuni succesive.

### 3. Coliziunile reale sunt neglijabile

0–9 per pereche cu infer_vector, față de 477–705 cu model.dv. Embedding-urile bazate pe conținut reflectă corect similaritatea semantică.

### 4. Tranziția 1.11 → 2.0 este cea mai „curată"

Doar **0.28% leakage** (2 fișiere), ceea ce corespunde cu restructurarea majoră a codebase-ului Django la trecerea de la seria 1.x la 2.x.

### 5. Leakage-ul crește cu acumularea versiunilor

De la ~12% (perechea 1) la ~15% (perechea 23). Cu cât acumulăm mai multe versiuni în training, cu atât crește probabilitatea ca un fișier din test să aibă un corespondent neschimbat într-una din versiunile anterioare.

### 6. Fișierele buggy au o rată de leakage mai mare

În versiunile târzii (3.1+), **20–31% din fișierele buggy** sunt leaked, față de 4–12% clean. Aceasta sugerează că fișierele cu defecte tind să fie mai stabile (mai puțin modificate) între versiuni — un rezultat cu potențiale implicații pentru predicția defectelor.

## Implicații pentru HRIA

- Rezultatele cu **model.dv trebuie excluse** din orice analiză — sunt invalidate de artefactul de realocare
- **infer_vector** este metoda corectă pentru analiza de duplicat/leakage cu Doc2Vec
- Costul: timp de execuție **~10x mai mare** (64 min vs 6 min), dar rezultatele sunt fiabile
- Leakage-ul real de **0.28–15%** la nivel de fișier confirmă existența contaminării train/test în CVDP, dar la un nivel semnificativ mai mic decât ce raportau rezultatele eronate cu model.dv
- Urmează analiza la **nivel de metodă** pentru a verifica dacă granularitatea mai fină reduce sau amplifică leakage-ul

## Reproducibilitate

```bash
# Rulare cu infer_vector (metoda corectă)
# GitHub Actions → Cross-Version Duplicate Analysis → Run workflow
# Parametri default: Django, threshold=0.99, epochs=10
```

Artefactele ambelor rulări sunt disponibile pe GitHub Actions (retenție 90 zile):
- [model.dv artifacts](https://github.com/ioanachelaru/doc2vec-experiments/actions/runs/33478789208)
- [infer_vector artifacts](https://github.com/ioanachelaru/doc2vec-experiments/actions/runs/33485534592)
