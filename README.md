# stbi-mmi24

How to run
pip install -r requirements.txt
docker compose up -d

python text_embedding.py // only need once

uvicorn main:app --reload

how to test
http://127.0.0.1:8000/search/Elasticsearch%20is
