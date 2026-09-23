# CareerLens AI

Intelligent Resume–Job Matching and Skill Gap Analysis System Using NLP and Machine Learning.

## Features

- PDF, DOCX and TXT resume parsing
- Resume category prediction
- TF-IDF job matching
- Weighted skill-gap analysis
- Top job recommendations
- Job-market analytics
- User registration and login
- Application tracker using SQLite
- Interactive Streamlit interface

## Run

1. Open `CareerLens_AI_Practical.ipynb` in VS Code.
2. Run every cell from top to bottom.
3. Confirm that the model and processed-data files are created.
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run streamlit_app.py`.

## Deployment Note

SQLite is suitable for the academic and local demonstration. Streamlit Community Cloud can reset local app storage when an app restarts or redeploys, so permanent public account storage should later be moved to a hosted SQL service.
