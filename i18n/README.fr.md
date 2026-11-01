<p align="center">
  <img src="../docs/assets/ascend-h3-logo.png" alt="Ascend-H3 logo" width="190">
</p>

<h1 align="center">Ascend-H3</h1>

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](../LICENSE)
[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue)](../docs/environment.md)
[![OS: Linux](https://img.shields.io/badge/OS-Linux-lightgrey)](../docs/environment.md)
[![Platform: Ascend 910B3](https://img.shields.io/badge/platform-Ascend%20910B3-0077C8)](../docs/validation.md)
[![Backend: vLLM-Omni](https://img.shields.io/badge/backend-vLLM--Omni-orange)](../docs/environment.md)

</div>

<p align="center"><a href="#install">Installation</a> | <a href="#models">Models</a> | <a href="#serve">Serve</a> | <a href="#performance">Performance</a></p>

<div align="center">

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | **Français** | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> Ce README complet est une traduction assistée par l'IA du README anglais au commit be5c42e03f9a. Il peut devenir obsolète ; consultez la version anglaise pour les dernières informations. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

Inférence MiniMax H3 sur Ascend avec attention complète, VSA, Veda, VDN et Spark-H3 dans un seul référentiel. L'intégration réutilise le chargement du modèle vLLM-Omni, la quantification du transformateur INT8 et les têtes d'attention parallèles tensorielles natives.

**Portée validée :** 40/40 cellules INT8 E2E à 1344 × 768, 124/345 images et 1/2/4/8 NPU (37 mesures fraîches et 3 cellules de déchargement de couche historiques explicitement marquées). La comparaison résidente BF16 sur Ascend 910B3 et PRO 6000 contient 20 mesures de base d'attention et 20 mesures de bloc complet. Les opérateurs d'attention conservent leurs versions validées.

## Charges de travail 768p par défaut

| Cas | Largeur × hauteur | Trames de sortie natives | Grille vidéo latente |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

Le FPS de sortie est de 24. Les requêtes utilisent un nombre de trames explicites, avec une interpolation de trames désactivée. Le cas long est la charge de travail native convenue de 345 frames ; il n'est pas recadré à 341 images. BF16 et INT8 utilisent ces deux charges de travail 1344×768. L’omission de `--case` sélectionne `1344-124`.

<a id="install"></a>

## Mise en place

Utilisez Linux, Python 3.12 et un environnement compatible Ascend Torch/CANN. Installez la pile vLLM épinglée, vLLM-Ascend et vLLM-Omni décrite dans [les notes d'environnement](../docs/environment.md). Ne combinez pas des versions de framework non liées. Installez ensuite ce référentiel à partir des sources :

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Les builds ne contiennent aucun poids de modèle et s'exécutent sans occuper de NPU. Les bibliothèques natives sont compilées localement et conservées en dehors de la caisse. Pour un seul backend, remplacez `all` par `vsa`, `veda` ou `spark` ; Full et VDN ne nécessitent pas ces extensions CANN personnalisées. Gardez la caisse modifiable disponible.

<a id="models"></a>

## Points de contrôle et choix d’algorithmes

| Méthode | Éléments de modèle requis | Calcul |
| --- | --- | --- |
| `full` | Piètement H3 ; LightX2V 8 étapes par défaut, ou une communauté LoRA | Attention intense |
| `vsa` | Étudiant officiel complet FastH3 V2, comprenant les 50 portes de compression formées | 8 avant, 80 % de rareté vidéo ; KV64 logique, blocs KV sélectionnés appariés en KV128 sur puce |
| `veda` | Base H3 + LightX2V 8 étapes + prédicteur Veda officiellement formé | Itinéraires de blocs dynamiques par tête Veda, parcimonie nominale de 90 % |
| `vdn` | Base H3 + répertoire officiel des points de contrôle VDN stage-DMD | Softmax local plus branche DeltaNet apprise à 128 dimensions |
| `spark` | Base H3 + LightX2V 8 étapes | Partition hiérarchique Spark-H3 et joint softmax exact/proxy, rareté nominale des branches exactes de 90 % |

Téléchargez les modèles sous leurs licences d'origine. Les poids ne sont pas redistribués. Différents étudiants distillés ont des horaires différents et des règles apprises ; accorder une attention particulière à des poids arbitraires ne garantit pas une qualité correspondante. VSA nécessite des portes entraînées différentes de zéro. VDN nécessite ses poids de branche appris. Les commandes de téléchargement et de préparation se trouvent dans [actifs du modèle](../docs/weights.md).

Full, Veda et Spark activent par défaut le **LightX2V LoRA** en huit étapes épinglé. Utilisez `--no-few-step-lora` pour exécuter le modèle original en plusieurs étapes. Une communauté préparée `--lora /models/style` est fusionnée avec LightX2V avant la fin du chargement ; désactivez l'étudiant par défaut lors de la sélection explicite d'un autre étudiant de la communauté. Veda a toujours besoin de son prédicteur officiel formé. VSA accepte uniquement le point de contrôle complet officiel complet **FastH3 V2** : huit avant, 80 % de parcimonie vidéo, décalages vidéo/audio 10/3, tuile 64, guidage 1. Il charge directement les poids libérés et ne prend pas `--lora`. VDN accepte uniquement son point de contrôle officiel complet Stage-DMD. Ni l’un ni l’autre n’accepte une communauté supplémentaire LoRA ou un indicateur de désactivation des étudiants.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

La source officielle PEFT est normalisée une fois en utilisant son alpha/rang déclaré ; les fichiers préparés sont vérifiés par hachage. Tous les adaptateurs sélectionnés fusionnent avant le découpage TP et la quantification INT8. Il n’y a pas de multiplications matricielles LoRA au moment de la requête. Voir [préparation du modèle](../docs/weights.md) pour les formats source et les planifications fixes.

<a id="serve"></a>

## Démarrer un serveur

Utilisez uniquement des appareils qui ne disposent d'aucun autre processus de calcul. Pour une carte :

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

D'autres backends utilisent la même API de serveur :

```bash
h3-ascend serve --method vsa --model /models/FastH3-V2 \
  --precision int8 --devices 1 --offload text
h3-ascend serve --method veda --model /models/MiniMax-H3 \
  --predictor /models/veda/predictor.safetensors \
  --precision int8 --devices 1 --offload text
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 --precision int8 --devices 1 --offload text
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Exécutez un serveur à la fois sur les appareils sélectionnés. Ajoutez `--dry-run` pour inspecter la commande native vLLM-Omni générée sans charger de modèle.

Les exemples à carte unique diffusent l'encodeur de texte tout en conservant le résident INT8 DiT. La recette native quantifie le DiT ; l'encodeur de texte reste BF16, son empreinte mémoire doit donc toujours être incluse dans la politique de lancement. Pour **BF16**, utilisez `--precision bf16` ; si nécessaire, sélectionnez explicitement `--offload layer --no-precompute-adaln`. INT8 utilise la recette native :

```json
{"transformer":{"method":"int8"}}
```

Les projections INT8 attirent l’attention sur BF16 Q/K/V. Attention les accumulateurs, les prédictions Veda et l'état VDN conservent leur précision prescrite. Il s'agit d'une inférence de modèle INT8, et non d'une affirmation selon laquelle chaque opération d'attention est INT8. La recette native conserve les couches non prises en charge dans BF16. Le H3 à carte unique a une projection de 96 768 de large qui dépasse la limite de sortie matmul quantifiée NPU de 65 535 ; TP peut réduire sa largeur par rang en portée.

Pour l'**inférence à quatre cartes**, utilisez le partitionnement de tête parallèle tenseur natif :

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` prend en charge 1, 2, 4, 7 et 8, qui divisent 56 têtes. Veda conserve les ID de tête de prédicteur global après le partitionnement. Le DiT TP à sept cartes utilise un seul rang d'encodeur de texte car ses 64 têtes de requête/8 KV ne peuvent pas être divisées par sept. Lance par défaut `--offload none` (poids résidents). Si la mémoire est insuffisante, `--offload text` diffuse uniquement l'encodeur de texte tout en conservant le DiT sur l'appareil ; `--offload module` échange des composants complets (qualification expérimentale) ; `--offload layer` diffuse les deux composants, en conservant un bloc actif et un bloc préextrait. La référence de déchargement correspondant complétée ci-dessous utilise explicitement `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) sur les deux plates-formes matérielles. Les nouvelles mesures de bout en bout utilisent **INT8 uniquement**, en commençant par l'exécution résidente et en marquant chaque repli de déchargement nécessaire. L'attention isolée et la comparaison des blocs DiT utilisent BF16 sur les deux appareils : couches 0/24/49, trois exécutions chacune, additionnant les neuf timings et divisant par neuf. Le `--parallel ulysses` facultatif est exposé pour les chemins non-VDN ; TP reste la valeur par défaut. Le décodage natif VAE en mosaïque est désormais par défaut sur toutes les cartes sélectionnées (`--vae-parallel-size 0`) ; utilisez `--vae-parallel-size 1` pour un seul rang de décodeur. Le pipeline INT8 Full / 124 images a obtenu un, deux, quatre et huit rangs VAE. Les autres combinaisons méthode/compte d’images restent dans la matrice E2E.

Les lancements de NPU résidents préparent les résultats AdaLN avant que le serveur ne soit prêt. La préparation résout le modèle chargé et fusionne les positions sigma réelles de LoRA, le nombre de mises à jour de DiT et les décalages vidéo/audio ; il n'est pas codé en dur en huit étapes. Utilisez `serve --steps N` pour spécifier un lancement enseignant/étudiant personnalisé et utilisez le même nombre dans `generate --steps N`. Un programme de points de contrôle distillé contradictoire est rejeté. La capacité du cache augmente avec les plans requis. Les calculs de projection d'origine sont conservés ; Les poids principaux AdaLN restent sur le CPU, tandis que les résultats préparés restent sur le NPU. TP4/TP8 mémorisent également les intégrations temporelles natives à partir d'entrées de pas de temps exacts pour éviter que la variation de réduction FP32 à bits faibles n'invalide les caches en aval. Les poids, les paramètres numériques et l’accord de tous les rangs protègent toujours la réutilisation. Redémarrez avec la nouvelle configuration pour précalculer un planning différent ; une entrée non préparée utilise le calcul d'origine et peut transférer des poids. `--no-precompute-adaln` désactive la préparation ; Le streaming de module/couche DiT le maintient désactivé par défaut.

Pour Spark/VSA/VDN à 345 images à carte unique, utilisez `--offload layer --no-precompute-adaln` avec les opérateurs validés d'origine. Le tableau E2E marque ces cellules de déchargement de couche native avec †L ; ‡R identifie les mesures correspondantes réutilisées lors de l'exécution précédente. Toutes les autres cellules conservent leur politique de résidence enregistrée individuellement.

Le lanceur définit le délai de demande du travailleur natif à 7200 secondes afin qu'une longue vidéo ne soit pas abandonnée par la valeur par défaut de 600 secondes en amont. Définissez `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` avant le lancement pour outrepasser cette limite ; cela modifie le comportement d'attente, pas le calcul du modèle ou la durée mesurée de la requête.

## Générer les deux cas 768p

Créez un fichier d'invite UTF-8, démarrez un serveur, puis utilisez le nombre de pas correspondant :

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Transmettez le `--method` du serveur au client : les cinq méthodes (Full/VSA/Veda/VDN/Spark) utilisent par défaut huit transferts réels. Pour le modèle original en plusieurs étapes en option, utilisez `generate --method full --no-few-step-lora` (50 étapes), correspondant au serveur. Un `--steps` explicite reste disponible pour un étudiant communautaire préparé séparément. `--guidance-scale` est facultatif ; l’omission préserve la politique autochtone. Aucune étape Affiner n’est ajoutée. Le client enregistre le MP4 et un reçu de synchronisation JSON.

<!-- BEGIN BF16 LAYER RESULTS -->

## Comparaison des couches BF16 : résident sur les deux appareils

1344×768, 124/345 images uniquement. Poids réels des étudiants et états cachés synthétiques gelés identiques aux octets ; ce ne sont pas des activations de couche capturées à partir d’une demande vidéo. Les couches 0, 24 et 49 reçoivent chacune deux échauffements non chronométrés et trois courses formelles : **sommez les neuf timings et divisez par neuf**. Les poids restent résidents sur les deux appareils. Les activations/poids principaux sont BF16 ; Les prédicteurs FP32 requis, les accumulations et l'état VDN sont conservés. Les accélérations utilisent le même matériel et la même forme de base BF16 Full. L'accélération relative plus élevée est en gras ; la latence absolue est également affichée.

### Attention de base uniquement

Commence par Q/K/V préparé et se termine à la sortie attention. Exclut la projection QKV, la norme Q/K, RoPE, la projection de sortie, AdaLN et FFN. Comprend les portes clairsemées requises, la prédiction, la sélection, le réarrangement et la restauration ; VDN comprend à la fois des noyaux softmax complets et linéaires. Chaque méthode utilise cette limite ; les timings de l’ensemble du module ne sont pas remplacés. La synchronisation du cœur a complété 10/10 paires matérielles correspondantes ; les cellules en attente ne réutilisent pas les mesures antérieures du module.

| Cadres | Méthode | Latence Ascend | Latence PRO 6000 | Accélération Ascend | Accélération PRO 6000 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 343.31 ms | 111.64 ms | 1.00× | 1.00× |
| 124 | VSA V2 · 80% | 216.24 ms | 93.56 ms | **1.59×** | 1.19× |
| 124 | Veda · 90% | 75.50 ms | 25.01 ms | **4.55×** | 4.46× |
| 124 | VDN · official | 368.89 ms | 142.74 ms | **0.93×** | 0.78× |
| 124 | Spark-H3 · 90% | 128.54 ms | 39.35 ms | 2.67× | **2.84×** |
| 345 | Full | 2578.60 ms | 838.10 ms | 1.00× | 1.00× |
| 345 | VSA V2 · 80% | 1196.30 ms | 498.34 ms | **2.16×** | 1.68× |
| 345 | Veda · 90% | 380.26 ms | 123.45 ms | 6.78× | **6.79×** |
| 345 | VDN · official | 1093.46 ms | 423.12 ms | **2.36×** | 1.98× |
| 345 | Spark-H3 · 90% | 636.98 ms | 184.38 ms | 4.05× | **4.55×** |

### Bloc complet DiT

S'exécute de l'entrée du bloc à la sortie du bloc, y compris les projections, la normalisation, RoPE, le noyau d'attention, AdaLN, FFN et les opérations résiduelles. Mesuré indépendamment du noyau ; leur différence n'est pas une mesure FFN, et aucune n'est extrapolée dans le temps E2E.

| Cadres | Méthode | Latence Ascend | Latence PRO 6000 | Accélération Ascend | Accélération PRO 6000 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 491.51 ms | 185.93 ms | 1.00× | 1.00× |
| 124 | VSA V2 · 80% | 346.40 ms | 166.56 ms | **1.42×** | 1.12× |
| 124 | Veda · 90% | 213.56 ms | 97.76 ms | **2.30×** | 1.90× |
| 124 | VDN · official | 527.75 ms | 222.89 ms | **0.93×** | 0.83× |
| 124 | Spark-H3 · 90% | 267.42 ms | 112.65 ms | **1.84×** | 1.65× |
| 345 | Full | 3000.60 ms | 1054.58 ms | 1.00× | 1.00× |
| 345 | VSA V2 · 80% | 1550.07 ms | 703.92 ms | **1.94×** | 1.50× |
| 345 | Veda · 90% | 787.91 ms | 330.60 ms | **3.81×** | 3.19× |
| 345 | VDN · official | 1534.19 ms | 647.19 ms | **1.96×** | 1.63× |
| 345 | Spark-H3 · 90% | 1028.43 ms | 393.06 ms | **2.92×** | 2.68× |

![Comparaison du noyau d'attention BF16 et du bloc DiT](../docs/figures/bf16-three-layer.png)[Tous les neuf enregistrements et contrôles d'identité d'entrée](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E : Ascend 1/2/4/8 cartes

1344 × 768, huit mises à jour réelles de DiT, y compris l'encodage de texte, le décodage vidéo/audio et l'écriture MP4. Le chargement, la fusion préalable de LoRA, la préparation et les échauffements de AdaLN sont exclus. Préférez le résident DiT ; les lignes résidentes vérifient zéro échec AdaLN et aucun transfert de poids sur chaque rang autour de la demande formelle. Après les MOO enregistrés, les cellules marquées de 345 images à carte unique utilisent le déchargement de couche native et la réutilisation correspondante, des mesures préalables entièrement validées comme demandé ; ces cellules sont explicitement marquées plutôt que présentées comme de nouvelles mesures.

| Cadres | Méthode | 1 carte | 2 cartes | 4 cartes | 8 cartes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 197.66 s †T | 116.03 s | 64.78 s | 35.87 s |
| 124 | VSA V2 · 80% | 146.81 s †T | 93.57 s | 53.47 s | 30.40 s |
| 124 | Veda · 90% | 97.40 s †T | 69.66 s | 43.38 s | 25.33 s |
| 124 | VDN · official | 216.91 s †T | 136.02 s | 79.26 s | 44.36 s |
| 124 | Spark-H3 · 90% | 117.11 s †T | 79.41 s | 48.87 s | 29.05 s |
| 345 | Full | 1211.17 s †T | 649.72 s †T | 328.63 s | 178.81 s |
| 345 | VSA V2 · 80% | 617.55 s †L ‡R | 363.99 s †T | 193.33 s | 104.71 s |
| 345 | Veda · 90% | 337.95 s †T | 225.31 s †T | 130.11 s | 75.93 s |
| 345 | VDN · official | 620.67 s †L ‡R | 388.58 s †T | 209.84 s | 117.27 s |
| 345 | Spark-H3 · 90% | 419.44 s †L ‡R | 267.61 s †T | 150.50 s | 87.02 s |

Couvert **40/40** : 37 mesures antérieures fraîches et 3 réutilisées. †T : déchargement de la couche d'encodeur de texte uniquement ; †L : déchargement de couche DiT/encodeur de texte natif avec prélecture à une couche et VAE résident ; ‡R : correspondant aux mesures originales de 2026-10-31. Les cellules non marquées n’ont pas de déchargement d’exécution. Seules les lignes résidentes précalculent AdaLN ; Les lignes de déchargement de couches ne font aucune réclamation de résidence DiT ou de transfert nul-AdaLN. VAE classe le nombre de cartes de correspondance ; Têtes DiT divisées 56/28/14/7. La couverture native INT8 est conservée.[Enregistrements E2E INT8](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

La comparaison couche-déchargement correspondante antérieure est conservée dans la [référence historique à carte unique](../docs/archive/matched-offload-single-card.md). Les cellules réutilisées dans le tableau actuel sont liées aux enregistrements d'origine inchangés.

## Résultats qualitatifs et validation

Ascend 910B3 uniquement : **1344 × 768, 345 images natives, 24 FPS (14,375 secondes, environ 14,4 secondes)**, avec audio.

Les cinq méthodes utilisent huit avant. À gauche : BF16. À droite : W8A8 natif INT8. Tous les échantillons utilisent une carte, la même invite, la graine commune **7** et le déchargement correspondant.

> Un cycliste roule le long d'un sentier verdoyant et calme, un travelling cinématique fluide, la lumière naturelle du jour.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V8 · 8 étapes**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Full + LightX2V8 · 8 étapes**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80 % · 8 étapes**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80 % · 8 étapes**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90 % · 8 étapes**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90 % · 8 étapes**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · officiel · 8 étapes**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · officiel · 8 étapes**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 étapes**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 étapes**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 utilise une rareté vidéo de 80 % ; Veda en utilise 90 %. Le texte et l'audio conservent leurs règles natives.[Paramètres, SHA256 et provenance des dix vidéos](../examples/videos/qualitative-1344-345.json). Le décodage complet vérifie 345 images et audio ; 16 images uniformes par clip ont été examinées visuellement. Les élèves distincts ne sont pas considérés comme étant équivalents en termes de pixels ou de perception. VSA conserve les changements de prise de vue ; aucune garantie sans scintillement n'est faite.

L'intégration initiale cible le premier lot T2VA sans vidéos de référence. Voir [architecture](../docs/architecture.md),[état de validation](../docs/validation.md) et [avis tiers](../THIRD_PARTY_NOTICES.md).
