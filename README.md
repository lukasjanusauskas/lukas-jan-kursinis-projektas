# Tradicinių ir giliojo mašininio mokymosi metodų lyginamoji analizė prognozių intervalais pagrįstam jūrų eismo anomalijų aptikimui Baltijos jūroje

**Darbą atliko: [Lukas Janušauskas](mailto:lukas.janusauskas@mif.stud.vu.lt)**

**Darbo vadovas: asist. dr. Julius Venskus**

---

## Turinys

- [Paleidimo instrukcijos](#paleidimo-instrukcijos)
- [Surinkti duomenys](#surinkti-duomenys)
- [Modeliavimas](#modeliavimas)
- [Analizė](#analizė)

---

## Paleidimo instrukcijos

Naudotų programų versijos:

1. **Python 3.8.5** (dėl suderinamumo su HPC) ir **Python 3.12.11** — skirtingiems failams naudojama skirtinga versija.
2. Leidžiant kodą per Python 3.8.5, reikia turėti įsidiegus `requirements-venv.txt` paketus.
3. Leidžiant kodą per Python 3.12.11, reikia turėti įsidiegus `requirements-dotvenv.txt` paketus.

> **Rekomendacija:** naudoti virtual environment arba conda-forge.

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Unix (macOS / Linux)

> Jei naudojate Ubuntu, gali tekti vietoje `python` rašyti `python3`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Surinkti duomenys

Kad negautumėte klaidų, sukurkite direktoriją `data/` projekto pagrindiniame aplanke (ne subdirektorijoje, o tiesiog projekto šaknyje).

Aktyvavę virtualią aplinką, paleiskite:

```bash
python -m src.duomenu_surinkimas
```

### Yi-Peng 3 anomalijų duomenys

Duomenų surinkimas aprašytas `notebooks/data-acquisition-yipeng3.ipynb` ir `notebooks/data-acquisition-yipeng3.ipynb` faile. Tam reikia:

- Yi-Peng 3 AIS duomenų failo (viešinti negaliu);
- Stormglass API rakto, leidžiančio atlikti bent **1 400 užklausų per dieną**.

Galutinius failus `anom-x.npy` ir `anom-y.npy` galiu atsiųsti asmeniškai — rašykite man.

### Meteorologiniai duomenys

Meteorologinių duomenų surinkimas atliekamas paleidžiant `src/meteorological.py`.

> Reikia Stormglass API rakto, leidžiančio bent **7 260 užklausų per dieną**. Jei paleisit *Yi-Peng 3* duomenų surinkimą, šitą galėsite paleisti tik kitą dieną. 

### Galutinis duomenų paruošimas

Paleiskite `src/data_preparation.py`. Jis sukurs šiuos failus:

- `X_train_final.npy`
- `y_train_final.npy`
- `X_test_final.npy`
- `y_test_final.npy`
- `X_val_final.npy`
- `y_val_final.npy`

---

## Modeliavimas

### LSTM-AE

| Failas | Aprašymas |
|--------|-----------|
| `src/lstm_ae_funkcijos.py` | LSTM architektūros funkcijos ir hiperparametrai |
| `src/lstm_ae_bandymas_hyper.py` | Hiperparametrų eksperimentas ir validavimo duomenų generavimas |

### QRF

| Žingsnis | Failas |
|----------|--------|
| 1. Hiperparametrų atranka | `src/qrf-bandymas-3.py` |
| 2. Kintamųjų atranka | `src/qrf_kintamuju_atrinkimas.py` |
| 3. Modelis su Yi-Peng 3 duomenimis | `src/qrf_kintamuju_atrinkimas-2.py` |
| 4. PI matų dinamikos tyrimas | `src/qrf_kintamuju_atrinkimas-3.py` |

---

## Analizė

| Failas | Aprašymas |
|--------|-----------|
| `notebooks/pradine-atvejo-analize.ipynb` | Pradinė atvejo analizė |
| `notebooks/duomenu-rinkinio-analize.ipynb` | Duomenų rinkinio analizė |
| `notebooks/pi-matai-qrf.ipynb` | PI matai QRF modeliui ištirti |
| `notebooks/anomalijos-patikra-qrf.ipynb` | QRF modelio Yi-Peng 3 anomalijos patikra |
| `notebooks/klaidu-analize-qrf.ipynb` | QRF klaidų analizė |
| `src/anomaly-patikra-lstm-ae.ipynb` | LSTM-AE anomalijos patikra ir PI matų dinamikos tyrimas |
| `notebooks/epochu-kreives.ipynb` | LSTM-AE epochų–nuostolių funkcijos kreivės |