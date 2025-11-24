# Open-Source-Map-Recognition-AI

## Overzicht

Dit project ontwikkelt een AI-systeem dat landen kan identificeren op basis van beelden afkomstig uit Google Street View of andere videostreams. Het systeem gebruikt een NVIDIA-GPU voor training en inference en combineert beeldverwerking met deep learning-modellen. De toepassing ondersteunt zowel statische afbeeldingen als live videostreams.

## Functionaliteit

* Classificatie van landen op basis van afbeeldingen of schermopnames.
* Ondersteuning voor meerdere invoerbronnen:

  * Geüploade afbeeldingen
  * Live screenshots
  * Externe URL's
  * Willekeurige lokale testafbeeldingen
* Twee modellen kunnen parallel worden gebruikt:

  * **Model A**: Fijn-afgesteld SigLIP-model (lokale checkpoint)
  * **Model B**: GeoGuessr-55 model
* Optionele beeldbewerking, zoals uitsnijden van het centrale beeldgedeelte.
* Realtime weergave van voorspellingen met top-k scores en grafische visualisaties.

## Architectuur

### Invoer

* Google Street View-beelden
* Live videostreams:

  * Geoguessr-stream
  * Google Street View
  * Camerafeed

### Verwerking

* Python
* PyTorch (modeltraining en inferentie)
* OpenCV (beeldverwerking)
* Streamlit (UI en dashboard)


### Hardware

* NVIDIA GPU

### Uitvoer

* Landlabels
* Probabilistische scores
* Statistische analyse van modelprestaties

## Installatie

### Vereisten

* Python 3.10 of hoger
* CUDA-compatibele NVIDIA GPU (optioneel, aanbevolen)
* Afhankelijkheden vermeld in `requirements.txt` (indien aanwezig)

### Installatiestappen

```bash
git clone <repository-url>
cd <project-folder>
pip install -r requirements.txt
```

## Gebruik

Start de Streamlit-applicatie:

```bash
streamlit run ShowCase.py
```

### Modusopties

1. **Upload image**
   Gebruiker levert een afbeelding aan voor classificatie.

2. **Live screenshot (local)**
   Het systeem maakt periodiek een screenshot en voert inferentie uit.

3. **Example image URL**
   Laadt en verwerkt een afbeelding vanaf een extern adres.

4. **Random Local Image (auto)**
   Selecteert automatisch een willekeurige testafbeelding uit een lokale dataset.

### Modellen

* **Model A**

  * Gebaseerd op `google/siglip-base-patch16-224`
  * Fijn-afgesteld op een lokale dataset
  * Laadpad configureerbaar in `MODEL_A_PATH`

* **Model B**

  * Extern model: `prithivMLmods/GeoGuessr-55`
  * Mapping voor 55 landen aanwezig in de code

Beide modellen kunnen afzonderlijk worden geactiveerd via de zijbalk.

## Code-structuur

Belangrijke componenten:

| Module / Functie                | Beschrijving                                   |
| ------------------------------- | ---------------------------------------------- |
| `load_model_A` / `load_model_B` | Laden van modellen via caching                 |
| `predict_A` / `predict_B`       | Inferentiefuncties met top-k output            |
| `get_random_local_image`        | Selecteren van lokale testafbeeldingen         |
| `render_predictions`            | UI-rendering en visualisatie                   |
| Streamlit UI-secties            | Besturing van modi, visualisatie en interactie |

## Datasetbronnen

* GeoGuessr-images dataset (Kaggle)
* SIGLIP2-gebaseerde modellen
* Lokale datasetfolders gestructureerd per land

## Onderzoeksvraag

Onderzocht wordt of een model gebaseerd op SigLIP2 in staat is landen accuraat te herkennen vanuit videostreams en hoe dit model presteert ten opzichte van bestaande systemen zoals GeoGuessr-55.

Subvragen omvatten:

* Vergelijking van modelprestaties
* Platformonafhankelijkheid
* Efficiënte integratie met gebruikersprocessen
* Vergelijking met menselijke prestaties

## Verwachte resultaten

* Werkend systeem dat landen voorspelt op basis van video- en beeldinvoer.
* Ondersteuning voor realtime inferentie.
* Statistische evaluatie van modelprestaties.

## Referenties

1. [Kaggle GeoGuessr images dataset](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)
2. [NVIDIA CUDA Toolkit documentatie](https://developer.nvidia.com/cuda-toolkit)
3. [NVIDIA AI-acceleratie documentatie](https://developer.nvidia.com/accelerate-ai-applications/get-started)
4. [GPU-installatiegids](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning)
5. [GeoGuessr-55 (HuggingFace)](https://huggingface.co/prithivMLmods/GeoGuessr-55)
6. [Google SigLIP2 model (HuggingFace)](https://huggingface.co/google/siglip2-base-patch16-224)
