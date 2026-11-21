# Lignée du champion

Chaque étape : le parent, les mutations appliquées, le score sur A (fitness) et sur B (sélection), puis les lignes de code qui changent. Les scores sont en bits par caractère (plus petit = mieux).

## Étape 0 : génome de départ (PPM-C interpolé)

- A = 2.0480, B = 1.7144

## Étape 1 : génome n°2 (parent n°0, génération 1)

- mutations : feuille, supprime
- A : 2.0480 -> 1.8781 (-0.1699) ; B = 1.6090

```diff
-    p = 0 * n[0] + 1.0 / n.shape[-1]
+    p = 0 + 1.0 / n.shape[-1]
-        p = np.where(N[k] > 0, div(n[k] + e * p, N[k] + e), p)
+        p = np.where(N[k] > 0, div(n[k] + e * p, N[k] + n[k]), p)
```

## Étape 2 : génome n°15 (parent n°2, génération 6)

- mutations : instruction, croise
- A : 1.8781 -> 1.8781 (+0.0000) ; B = 1.6090

```diff
+        b = n[k] + n[k]
```

## Étape 3 : génome n°22 (parent n°15, génération 8)

- mutations : post, insere
- A : 1.8781 -> 1.7778 (-0.1003) ; B = 1.5622

```diff
-        b = n[k] + n[k]
+        b = n[k] + n[k] + n[k]
+    p = p ** 0.747
```

## Étape 4 : génome n°32 (parent n°22, génération 11)

- mutations : insere
- A : 1.7778 -> 1.7778 (+0.0000) ; B = non évalué

```diff
-        b = n[k] + n[k] + n[k]
+        b = div(n[k] + n[k] + n[k], p)
```

## Étape 5 : génome n°48 (parent n°32, génération 16)

- mutations : instruction, supprime
- A : 1.7778 -> 1.7778 (+0.0000) ; B = non évalué

```diff
-        b = div(n[k] + n[k] + n[k], p)
+        b = div(n[k], p)
+        q = u1[k] * e
```

## Étape 6 : génome n°62 (parent n°48, génération 19)

- mutations : perturbe, supprime
- A : 1.7778 -> 1.6682 (-0.1096) ; B = 1.4923

```diff
-        e = d[k] * 1.0
+        e = d[k] * 2.22581
-        p = np.where(N[k] > 0, div(n[k] + e * p, N[k] + n[k]), p)
+        p = np.where(N[k] > 0, div(n[k] + e * p, N[k]), p)
```

## Étape 7 : génome n°139 (parent n°62, génération 37)

- mutations : perturbe, insere
- A : 1.6682 -> 1.6915 (+0.0232) ; B = 1.5494

```diff
-        e = d[k] * 2.22581
+        e = d[k] * 2.22581 + u1[k]
-    p = p ** 0.747
+    p = p ** 0.5859
```

## Étape 8 : génome n°1104 (parent n°139, génération 225)

- mutations : croise, enveloppe
- A : 1.6915 -> 1.6915 (+0.0000) ; B = 1.5494

```diff
-    p = 0 + 1.0 / n.shape[-1]
+    p = np.sqrt(0) + 1.0 / n.shape[-1]
-        q = u1[k] * e
+        q = p
```

## Étape 9 : génome n°1196 (parent n°1104, génération 241)

- mutations : perturbe
- A : 1.6915 -> 1.6915 (+0.0000) ; B = non évalué

```diff
-        p = np.where(N[k] > 0, div(n[k] + e * p, N[k]), p)
+        p = np.where(N[k] > 0.25491, div(n[k] + e * p, N[k]), p)
```

## Étape 10 : génome n°1471 (parent n°1196, génération 290)

- mutations : perturbe, instruction
- A : 1.6915 -> 1.7489 (+0.0575) ; B = non évalué

```diff
+        m = n[k] + d[k]
-        p = np.where(N[k] > 0.25491, div(n[k] + e * p, N[k]), p)
-    p = p ** 0.5859
+        p = np.where(N[k] > 0.25491, div(n[k] + m * p, N[k]), p)
+    p = p ** 0.64919
```

## Étape 11 : génome n°1673 (parent n°1471, génération 331)

- mutations : feuille
- A : 1.7489 -> 1.7489 (+0.0000) ; B = non évalué

```diff
-        b = div(n[k], p)
+        b = div(u2[k], p)
```

## Étape 12 : génome n°2022 (parent n°1673, génération 397)

- mutations : enveloppe, perturbe
- A : 1.7489 -> 1.7489 (-0.0000) ; B = non évalué

```diff
-    p = np.sqrt(0) + 1.0 / n.shape[-1]
+    p = np.sqrt(0) + 0.92546 / n.shape[-1]
-        p = np.where(N[k] > 0.25491, div(n[k] + m * p, N[k]), p)
+        p = np.where(N[k] > 0.25491, div(np.maximum(n[k], 0.0) + m * p, N[k]), p)
```

## Étape 13 : génome n°3321 (parent n°2022, génération 641)

- mutations : insere, par_ordre
- A : 1.7489 -> 1.6276 (-0.1213) ; B = 1.4772

```diff
-        e = d[k] * 2.22581 + u1[k]
+        e = d[k] * [2.17814, 2.23475, 2.3698, 2.43409, 2.36564, 2.30956, 2.34621, 2.18404, 1.94839, 2.20264, 2.19865][k] + u1[k]
-        p = np.where(N[k] > 0.25491, div(np.maximum(n[k], 0.0) + m * p, N[k]), p)
+        p = np.where(N[k] > 0.25491, div(np.maximum(n[k], 0.0) + (e + m) * p, N[k]), p)
```

