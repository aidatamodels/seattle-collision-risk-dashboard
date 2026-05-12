# Exécution locale

Placez ces fichiers dans le même dossier :

- `build_dashboard_data.py`
- `index.html`
- `requirements.txt`
- `Data-Collisions.csv` (facultatif, mais recommandé pour éviter de dépendre d'Internet)

Installez les dépendances :

```bash
pip install -r requirements.txt
```

Générez le JSON :

```bash
python build_dashboard_data.py
```

Le script produit :

```text
dashboard_data.json
```

Lancez un serveur local pour tester le dashboard :

```bash
python -m http.server 8000
```

Ouvrez ensuite :

```text
http://localhost:8000
```
