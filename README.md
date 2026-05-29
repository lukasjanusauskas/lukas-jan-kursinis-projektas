# Tradicinių ir giliojo mašininio mokymosi metodų lyginamoji analizė prognozių intervalais pagrįstam jūrų eismo anomalijų aptikimui Baltijos jūroje

**Darbą atliko: Lukas Janušauskas (lukas.janusauskas@mif.stud.vu.lt)**
**Darbo vadovas: asist. dr. Julius Venskus**

## Paleidimo instrukcijos

Naudotų programų versijos:
1. Python 3.8.5 (dėl suderinamumo su HPC) ir Python 3.12.11. Skirtingiems failams skirtinga versija.
2. Leidžiant kodą per Python 3.8.5, reikia turėti įsidiegus `requirements-venv.txt`.
3. Leidžiant kodą per Python 3.12.11, reikia turėti įsidiegus `requirements-dotvenv.txt` paketus.
3. Leidžiant kodą per Python 3.8.5, reikia turėti įsidiegus `requirements-venv.txt` paketus.

> Rekomendacija: naudoti virtual environment arba conda-forge.

## Surinkti duomenis
---

Kad negautumėte klaidų, sukurkite direktoriją `data/` projekto pagrindiniame aplankale (ta prasme nekišti į subdirektorijas, o tiesiog projekto aplanke).
Aktyvavę virtualią aplinką (paruošę Python aplinką), paleiskite taip:

```powershell
python -m src.duomenu_surinkimas
```

- Leidžiant Unix operacinėse sistemose (MacOS, Linux):

> Jie naudojate Ubuntu jums gali tekti leisti ne `python`, bet `python3`

```bash
python -m venv vevn
source venv/bin/activate
pip install -r requirements.txt
```

Yi-Peng 3 anomalijos duomenų surinkimas yra atliktas `notebooks/data-acquisition-yipeng3.ipynb` faile.
Tam reikia turėti: Yi-peng 3 AIS duomenų failą, kurio viešinti negaliu; ir stormglass API key, galinčio atlikti bent 1400 užklausų per dieną.
Galutinius failus: `anom-x.npy` ir `anom-y.npy` galėsiu atsiųsti, rašyti man.

Meteorologinių duomenų surinkimas atlikemas, paleidžiant `meteorological.py`.
Deja, Jam paleisti reikia turėti API raktą stormglass paskirai, turinčiai bent 7260 užklausų per dieną leidimą.

Duomenų galutiniam paruošimui reikia paleisti `src/data_preparation.py` failą.
Jis sukurs failus:

- `X_train_final.npy`
- `y_train_final.npy`
- `X_test_final.npy`
- `y_test_final.npy`
- `X_val_final.npy`
- `y_val_final.npy`

## Modeliavimas
---

Aprašytos funkcijos, nusakančios LSTM architektūrą ir šio neuroninio tinklo parinktus hiperparametrus, faile `src/lstm_ae_funkcijos.py`.

Modelio hiperparametrų eksperimentas ir validavimo duomenų generavimas aprašytas faile `src/lstm_ae_bandymas_hyper.py`.

QRF modelio:

1. Hiperparametrų atranka vykdyta `src/qrf-bandymas-3.py`
2. Kintamųjų atranka vykdyta `src/qrf_kintamuju_atrinkimas.py`
3. Modelis paleistas ant Yi-Peng 3 duomenų - `src/qrf_kintamuju_atrinkimas-2.py`
3. PI matų dinamikos tyrimui matus apskaičiavau `src/qrf_kintamuju_atrinkimas-3.py`

## Analizė
---

Pradinė atvejo analizė vykdyta faile `notebooks/pradine-atvejo-analize.ipynb`.
Duomenų analizė atlikta faile `notebooks/duomenu-rinkinio-analize.ipynb`.
PI matai QRF modeliui ištirti `notebooks/pi-matai-qrf.ipynb`.
Anomalijos patikra (patikrinu, ar QRF modelis pagauna Yi-Peng 3 anomaliją) `notebooks/anomalijos-patikra-qrf.ipynb`.
QRF klaidų analizė atlikta `notebooks/klaidu-analize-qrf.ipynb`.
LSTM-AE modelio anomalijos patikra ir PI matų dinamikos tyrimas pateikti `src/anomaly-patikra-lstm-ae.ipynb`.
LSTM-AE modelio epochų-nuostolių funkcijos kreivės buvo nubrėžtos faile `notebooks/epochu-kreives.ipynb`.