# IBM Data Science Capstone — SpaceX Falcon 9 Landing Prediction

**Project status:** Independent, reproducible reconstruction of the IBM SpaceX capstone, **not** a copy of the learner's original completed lab notebooks. Historical sample: 2010–2020.

## Files

| File | Description |
| --- | --- |
| [SpaceX_Capstone_Reproducible.ipynb](SpaceX_Capstone_Reproducible.ipynb) | Executed notebook containing EDA, SQL on the compact sample, plots, Folium map, classification models and evaluation. Saved outputs included. |
| [SpaceX_SQL_Lab_Queries.ipynb](SpaceX_SQL_Lab_Queries.ipynb) | SQL lab notebook for the distinct 101-row missions dataset; **executable template, not executed with saved outputs**. |
| [SpaceX_Dash.py](SpaceX_Dash.py) | Interactive Plotly Dash dashboard, with launch-site and payload-range filters. Code prepared but **not live-tested here**. |
| [analyze_capstone.py](analyze_capstone.py) | Reproducible analysis script that regenerates plots, model metrics and the Folium HTML. Requires Basemap in addition to the packages below. |
| [spacex_model_compact.csv](spacex_model_compact.csv) | Compact 90-flight dataset used by the executed notebook and dashboard; payload rounded to nearest kilogram. |
| [Spacex.csv](Spacex.csv) | Separate historical 101-mission dataset for the SQL lab. |
| [Falcon9_Launch_Sites_Folium.html](Falcon9_Launch_Sites_Folium.html) | Generated interactive launch-site map. |
| [results.json](results.json) | Machine-readable summary of the independent analysis. |

## Reproduce

1. Clone the repository and open this project directory:
   ```bash
   git clone https://github.com/Diego078/Temas-de-Economia.git
   cd Temas-de-Economia/proyectos/ibm-data-science-capstone-spacex
   ```
2. Install basic dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn folium plotly dash notebook
   ```
3. Run `jupyter notebook`, then open `SpaceX_Capstone_Reproducible.ipynb` or `SpaceX_SQL_Lab_Queries.ipynb`.
4. To run the dashboard: `python SpaceX_Dash.py` and open `http://127.0.0.1:8050`.
5. For `analyze_capstone.py`, also install `basemap`. The script writes charts into an `assets/` directory.

## Data provenance and limitations

- Public dataset reference for the 90-flight analysis: [liam-stuart/Applied-Data-Science-Capstone — dataset_part_2.csv](https://github.com/liam-stuart/Applied-Data-Science-Capstone/blob/main/dataset_part_2.csv).
- Public SQL lab source: [liam-stuart/Applied-Data-Science-Capstone — Spacex.csv](https://github.com/liam-stuart/Applied-Data-Science-Capstone/blob/main/Spacex.csv).
- The two datasets cover **different populations** (90 modeled launches versus 101 historical missions). Do not mix their denominators.
- The project is for **historical coursework**, not current SpaceX landing reliability or operational forecasting.
- Dashboard figures in the associated presentation are reconstructed previews, not screenshots of a live running Dash session.
