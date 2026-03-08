# Luko Janušausko kursinis projektas
## Tradicinių ir giliojo mašininio mokymosi metodų lyginamoji analizė prognozių intervalais pagrįstam jūrų eismo anomalijų aptikimui Baltijos jūroje

Naudotų programų versijos:
1. python 3.8.5 (dėl suderinamumo su HPC)
2. Kitų programų/paketų versijos yra pateiktos `requirements.txt`

Rekomendacijos paleidimui:

- Leidžiant Windows OS:

Pirmą kartą leidžiant:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1 # Arba activate.bat jei leidžiame per Command Prompt
pip install  -r requirements.txt
```

- Leidžiant Unix operacinėse sistemose (MacOS, Linux):

> Jie naudojate Ubuntu jums gali tekti leisti ne `python`, bet `python3`

```bash
python -m venv vevn
source venv/bin/activate
pip install -r requirements.txt
```
